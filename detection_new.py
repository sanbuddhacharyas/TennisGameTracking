import os
import time

from PIL import Image, ImageDraw
import tensorflow as tf

import cv2
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

from detection import DetectionModel, center_of_box
from pose import PoseExtractor
from smooth import Smooth
from ball_detection import BallDetector
from statistics import Statistics
from stroke_recognition import ActionRecognition
from utils import get_video_properties, get_dtype, get_stickman_line_connection
from court_detection import CourtDetector
import matplotlib.pyplot as plt

import pandas as pd
import queue

from sklearn.linear_model  import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from tennis_bounce import create_model, segment_ball_trajectory, find_bouncing_point, keypoint_to_heatmap

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        

def get_stroke_predictions(video_path, stroke_recognition, strokes_frames, player_boxes):
    """
    Get the stroke prediction for all sections where we detected a stroke
    """
    predictions = {}
    cap = cv2.VideoCapture(video_path)
    fps, length, width, height = get_video_properties(cap)
    video_length = 2
    # For each stroke detected trim video part and predict stroke
    for frame_num in strokes_frames:
        # Trim the video (only relevant frames are taken)
        starting_frame = max(0, frame_num - int(video_length * fps * 2 / 3))
        cap.set(1, starting_frame)
        i = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            stroke_recognition.add_frame(frame, player_boxes[starting_frame + i])
            i += 1
            if i == int(video_length * fps):
                break
        # predict the stroke
        probs, stroke = stroke_recognition.predict_saved_seq()
        predictions[frame_num] = {'probs': probs, 'stroke': stroke}
    cap.release()
    return predictions


def find_strokes_indices(player_1_boxes, player_2_boxes, ball_positions, skeleton_df, verbose=0):
    """
    Detect strokes frames using location of the ball and players
    """
    ball_x, ball_y = ball_positions[:, 0], ball_positions[:, 1]
    smooth_x = signal.savgol_filter(ball_x, 3, 2)
    smooth_y = signal.savgol_filter(ball_y, 3, 2)

    # Ball position interpolation
    x = np.arange(0, len(smooth_y))
    indices = [i for i, val in enumerate(smooth_y) if np.isnan(val)]
    x = np.delete(x, indices)
    y1 = np.delete(smooth_y, indices)
    y2 = np.delete(smooth_x, indices)
    ball_f2_y = interp1d(x, y1, kind='cubic', fill_value="extrapolate")
    ball_f2_x = interp1d(x, y2, kind='cubic', fill_value="extrapolate")
    xnew = np.linspace(0, len(ball_y), num=len(ball_y), endpoint=True)

    if verbose:
        plt.plot(np.arange(0, len(smooth_y)), smooth_y, 'o', xnew,
                 ball_f2_y(xnew), '-r')
        plt.legend(['data', 'inter'], loc='best')
        plt.show()

    # Player 2 position interpolation
    player_2_centers = np.array([center_of_box(box) for box in player_2_boxes])
    player_2_x, player_2_y = player_2_centers[:, 0], player_2_centers[:, 1]
    player_2_x = signal.savgol_filter(player_2_x, 3, 2)
    player_2_y = signal.savgol_filter(player_2_y, 3, 2)
    x = np.arange(0, len(player_2_y))
    indices = [i for i, val in enumerate(player_2_y) if np.isnan(val)]
    x = np.delete(x, indices)
    y1 = np.delete(player_2_y, indices)
    y2 = np.delete(player_2_x, indices)
    player_2_f_y = interp1d(x, y1, fill_value="extrapolate")

    player_2_f_x = interp1d(x, y2, fill_value="extrapolate")
    xnew = np.linspace(0, len(player_2_y), num=len(player_2_y), endpoint=True)

    if verbose:
        plt.plot(np.arange(0, len(player_2_y)), player_2_y, 'o', xnew, player_2_f_y(xnew), '--g')
        plt.legend(['data', 'inter_cubic', 'inter_lin'], loc='best')
        plt.show()

    coordinates = ball_f2_y(xnew)
    # Find all peaks of the ball y index
    peaks, _ = find_peaks(coordinates)
    if verbose:
        plt.plot(coordinates)
        plt.plot(peaks, coordinates[peaks], "x")
        plt.show()

    neg_peaks, _ = find_peaks(coordinates * -1)
    if verbose:
        plt.plot(coordinates)
        plt.plot(neg_peaks, coordinates[neg_peaks], "x")
        plt.show()

    # Get bottom player wrists positions
    left_wrist_index = 9
    right_wrist_index = 10
    skeleton_df = skeleton_df.fillna(-1)
    left_wrist_pos = skeleton_df.iloc[:, [left_wrist_index, left_wrist_index + 15]].values
    right_wrist_pos = skeleton_df.iloc[:, [right_wrist_index, right_wrist_index + 15]].values

    dists = []
    # Calculate dist between ball and bottom player
    for i, player_box in enumerate(player_1_boxes):
        if player_box[0] is not None:
            player_center = center_of_box(player_box)
            ball_pos = np.array([ball_f2_x(i), ball_f2_y(i)])
            box_dist = np.linalg.norm(player_center - ball_pos)
            right_wrist_dist, left_wrist_dist = np.inf, np.inf
            if right_wrist_pos[i, 0] > 0:
                right_wrist_dist = np.linalg.norm(right_wrist_pos[i, :] - ball_pos)
            if left_wrist_pos[i, 0] > 0:
                left_wrist_dist = np.linalg.norm(left_wrist_pos[i, :] - ball_pos)
            dists.append(min(box_dist, right_wrist_dist, left_wrist_dist))
        else:
            dists.append(None)
    dists = np.array(dists)

    dists2 = []
    # Calculate dist between ball and top player
    for i in range(len(player_2_centers)):
        ball_pos = np.array([ball_f2_x(i), ball_f2_y(i)])
        box_center = np.array([player_2_f_x(i), player_2_f_y(i)])
        box_dist = np.linalg.norm(box_center - ball_pos)
        dists2.append(box_dist)
    dists2 = np.array(dists2)

    strokes_1_indices = []
    # Find stroke for bottom player by thresholding the dists
    for peak in peaks:
        player_box_height = max(player_1_boxes[peak][3] - player_1_boxes[peak][1], 130)
        if dists[peak] < (player_box_height * 4 / 5):
            strokes_1_indices.append(peak)

    strokes_2_indices = []
    # Find stroke for top player by thresholding the dists
    for peak in neg_peaks:
        if dists2[peak] < 100:
            strokes_2_indices.append(peak)

    # Assert the diff between to consecutive strokes is below some threshold
    while True:
        diffs = np.diff(strokes_1_indices)
        to_del = []
        for i, diff in enumerate(diffs):
            if diff < 40:
                max_in = np.argmax([dists[strokes_1_indices[i]], dists[strokes_1_indices[i + 1]]])
                to_del.append(i + max_in)

        strokes_1_indices = np.delete(strokes_1_indices, to_del)
        if len(to_del) == 0:
            break

    # Assert the diff between to consecutive strokes is below some threshold
    while True:
        diffs = np.diff(strokes_2_indices)
        to_del = []
        for i, diff in enumerate(diffs):
            if diff < 40:
                max_in = np.argmax([dists2[strokes_2_indices[i]], dists2[strokes_2_indices[i + 1]]])
                to_del.append(i + max_in)

        strokes_2_indices = np.delete(strokes_2_indices, to_del)
        if len(to_del) == 0:
            break

    # Assume bounces frames are all the other peaks in the y index graph
    bounces_indices = [x for x in peaks if x not in strokes_1_indices]
    if verbose:
        plt.figure()
        plt.plot(coordinates)
        plt.plot(strokes_1_indices, coordinates[strokes_1_indices], "or")
        plt.plot(strokes_2_indices, coordinates[strokes_2_indices], "og")
        plt.legend(['data', 'player 1 strokes', 'player 2 strokes'], loc='best')
        plt.show()

    return strokes_1_indices, strokes_2_indices, bounces_indices, player_2_f_x, player_2_f_y


def mark_player_box(frame, boxes, frame_num):
    box = boxes[frame_num]
    if box[0] is not None:
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [255, 0, 255], 2)
    return frame


def mark_skeleton(skeleton_df, img, img_no_frame, frame_number):
    """
    Mark the skeleton of the bottom player on the frame
    """
    # landmarks colors
    circle_color, line_color = (0, 0, 255), (255, 0, 0)
    stickman_pairs = get_stickman_line_connection()

    skeleton_df = skeleton_df.fillna(-1)
    values = np.array(skeleton_df.values[frame_number], int)
    points = list(zip(values[5:17], values[22:]))
    # draw key points
    for point in points:
        if point[0] >= 0 and point[1] >= 0:
            xy = tuple(np.array([point[0], point[1]], int))
            cv2.circle(img, xy, 2, circle_color, 2)
            cv2.circle(img_no_frame, xy, 2, circle_color, 2)

    # Draw stickman
    for pair in stickman_pairs:
        partA = pair[0] - 5
        partB = pair[1] - 5
        if points[partA][0] >= 0 and points[partA][1] >= 0 and points[partB][0] >= 0 and points[partB][1] >= 0:
            cv2.line(img, points[partA], points[partB], line_color, 1, lineType=cv2.LINE_AA)
            cv2.line(img_no_frame, points[partA], points[partB], line_color, 1, lineType=cv2.LINE_AA)
    return img, img_no_frame


def add_data_to_video(input_video, court_detector, players_detector, ball_detector, strokes_predictions, skeleton_df,
                      statistics,
                      show_video, with_frame, output_folder, output_file, p1, p2, f_x, f_y):
    """
    Creates new videos with pose stickman, face landmarks and blinks counter
    :param input_video: str, path to the input videos
    :param df: DataFrame, data of the pose stickman positions
    :param show_video: bool, display output videos while processing
    :param with_frame: int, output videos includes the original frame with the landmarks
    (0 - only landmarks, 1 - original frame with landmarks, 2 - original frame with landmarks and only
    landmarks (side by side))
    :param output_folder: str, path to output folder
    :param output_file: str, name of the output file
    :return: None
    """

    player1_boxes = players_detector.player_1_boxes
    player2_boxes = players_detector.player_2_boxes

    player1_dists = statistics.bottom_dists_array
    player2_dists = statistics.top_dists_array

    if skeleton_df is not None:
        skeleton_df = skeleton_df.fillna(-1)

    # Read videos filequeue
    cap = cv2.VideoCapture(input_video)

    # videos properties
    fps, length, width, height = get_video_properties(cap)

    final_width = width * 2 if with_frame == 2 else width

    # Video writer
    out = cv2.VideoWriter(os.path.join(output_folder, output_file + '.avi'),
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (final_width, height))

    # initialize frame counters
    frame_number = 0
    orig_frame = 0
    while True:
        orig_frame += 1
        print('Creating new videos frame %d/%d  ' % (orig_frame, length), '\r', end='')
        if not orig_frame % 100:
            print('')
        ret, img = cap.read()

        if not ret:
            break

        # initialize frame for landmarks only
        img_no_frame = np.ones_like(img) * 255

        # add Court location
        img = court_detector.add_court_overlay(img, overlay_color=(0, 0, 255), frame_num=frame_number)
        img_no_frame = court_detector.add_court_overlay(img_no_frame, overlay_color=(0, 0, 255), frame_num=frame_number)

        # add players locations
        img = mark_player_box(img, player1_boxes, frame_number)
        img = mark_player_box(img, player2_boxes, frame_number)
        img_no_frame = mark_player_box(img_no_frame, player1_boxes, frame_number)
        img_no_frame = mark_player_box(img_no_frame, player2_boxes, frame_number)

        # add ball location
        img = ball_detector.mark_positions(img, frame_num=frame_number)
        img_no_frame = ball_detector.mark_positions(img_no_frame, frame_num=frame_number, ball_color='black')

        # add pose stickman
        if skeleton_df is not None:
            img, img_no_frame = mark_skeleton(skeleton_df, img, img_no_frame, frame_number)

        # Add stroke prediction
        for i in range(-10, 10):
            if frame_number + i in strokes_predictions.keys():
                '''cv2.putText(img, 'STROKE HIT', (200, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255) if i != 0 else (255, 0, 0), 3)'''

                probs, stroke = strokes_predictions[frame_number + i]['probs'], strokes_predictions[frame_number + i][
                    'stroke']
                cv2.putText(img, 'Forehand - {:.2f}, Backhand - {:.2f}, Service - {:.2f}'.format(*probs),
                            (70, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(img, f'Stroke : {stroke}',
                            (int(player1_boxes[frame_number][0]) - 10, int(player1_boxes[frame_number][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                break
        # Add stroke detected
        for i in range(-5, 10):
            '''if frame_number + i in p1:
                cv2.putText(img, 'Stroke detected', (int(player1_boxes[frame_number][0]) - 10, int(player1_boxes[frame_number][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255) if i != 0 else (255, 0, 0), 2)'''

            if frame_number + i in p2:
                cv2.putText(img, 'Stroke detected',
                            (int(f_x(frame_number)) - 30, int(f_y(frame_number)) - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255) if i != 0 else (255, 0, 0), 2)

        cv2.putText(img, 'Distance: {:.2f} m'.format(player1_dists[frame_number] / 100),
                    (50, 500),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(img, 'Distance: {:.2f} m'.format(player2_dists[frame_number] / 100),
                    (100, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # display frame
        if show_video:
            cv2.imshow('Output', img)
            if cv2.waitKey(1) & 0xff == 27:
                cv2.destroyAllWindows()

        # save output videos
        if with_frame == 0:
            final_frame = img_no_frame
        elif with_frame == 1:
            final_frame = img
        else:
            final_frame = np.concatenate([img, img_no_frame], 1)
        out.write(final_frame)
        frame_number += 1
    print('Creating new video frames %d/%d  ' % (length, length), '\n', end='')
    print(f'New videos created, file name - {output_file}.avi')
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# def create_top_view(court_detector, detection_model):
#     """
#     Creates top view video of the gameplay
#     """
#     court = court_detector.court_reference.court.copy()
#     court = cv2.line(court, *court_detector.court_reference.net, 255, 5)
#     v_width, v_height = court.shape[::-1]
#     court = cv2.cvtColor(court, cv2.COLOR_GRAY2BGR)
#     out = cv2.VideoWriter('output/top_view.avi',
#                           cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (v_width, v_height))
#     # players location on court
#     smoothed_1, smoothed_2 = detection_model.calculate_feet_positions(court_detector)

#     for feet_pos_1, feet_pos_2 in zip(smoothed_1, smoothed_2):
#         frame = court.copy()
#         frame = cv2.circle(frame, (int(feet_pos_1[0]), int(feet_pos_1[1])), 10, (0, 0, 255), 15)
#         if feet_pos_2[0] is not None:
#             frame = cv2.circle(frame, (int(feet_pos_2[0]), int(feet_pos_2[1])), 10, (0, 0, 255), 15)
#         out.write(frame)
#     out.release()
#     cv2.destroyAllWindows()

def diff_xy(coords):
  coords = coords.copy()
  diff_list = []
  for i in range(0, len(coords)-1):
    if coords[i] is not None and coords[i+1] is not None:
      point1 = coords[i]
      point2 = coords[i+1]
      diff = [abs(point2[0] - point1[0]), abs(point2[1] - point1[1])]
      diff_list.append(diff)
    else:
      diff_list.append(None)
  
  xx, yy = np.array([x[0] if x is not None else np.nan for x in diff_list]), np.array([x[1] if x is not None else np.nan for x in diff_list])
  
  return xx, yy

def remove_outliers(x, y, coords):
  ids = set(np.where(x > 50)[0]) & set(np.where(y > 50)[0])
  for id in ids:
    left, middle, right = coords[id-1], coords[id], coords[id+1]
    if left is None:
      left = [0]
    if  right is None:
      right = [0]
    if middle is None:
      middle = [0]
    MAX = max(map(list, (left, middle, right)))
    if MAX == [0]:
      pass
    else:
      try:
        coords[coords.index(tuple(MAX))] = None
      except ValueError:
        coords[coords.index(MAX)] = None

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def interpolation(coords):
    coords =coords.copy()
    x, y = [x[0] if x is not None else np.nan for x in coords], [x[1] if x is not None else np.nan for x in coords]

    xxx = np.array(x) # x coords
    yyy = np.array(y) # y coords

    nons, yy = nan_helper(xxx)
    xxx[nons]= np.interp(yy(nons), yy(~nons), xxx[~nons])
    nans, xx = nan_helper(yyy)
    yyy[nans]= np.interp(xx(nans), xx(~nans), yyy[~nans])

    newCoords = [*zip(xxx,yyy)]

    return newCoords

def create_top_view(court_detector, detection_model, coords, fps, tennis_tracking):
    """
    Creates top view video of the gameplay
    """
    # coords = xy[:]
    # Remove Outliers 
    x, y = diff_xy(coords)
    remove_outliers(x, y, coords)
    # Interpolation
    coords = interpolation(coords)

    court = court_detector.court_reference.court.copy()
    print(court, court.shape)
    court = cv2.line(court, *court_detector.court_reference.net, 255, 5)
   
    v_width, v_height = court.shape[::-1]
    court = cv2.cvtColor(court, cv2.COLOR_GRAY2BGR)
    out = cv2.VideoWriter('./minimap.mp4',cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, (v_width, v_height))
    # players location on court
    smoothed_1, smoothed_2 = detection_model.calculate_feet_positions(court_detector)
    i = 0 
    for feet_pos_1, feet_pos_2 in zip(smoothed_1, smoothed_2):
        frame = court.copy()
        frame = cv2.circle(frame, (int(feet_pos_1[0]), int(feet_pos_1[1])), 45, (255, 0, 0), -1)
        if feet_pos_2[0] is not None:
            frame = cv2.circle(frame, (int(feet_pos_2[0]), int(feet_pos_2[1])), 45, (255, 0, 0), -1)
        draw_ball_position(frame, court_detector, coords[i], i)

        tennis_tracking = tennis_tracking.append({"Time": (i / fps), "Frame":i, "Player_1":(int(feet_pos_1[0]), int(feet_pos_1[1])), \
            "Player_2":(int(feet_pos_2[0]), int(feet_pos_2[1])), "Ball_POS":coords[i], "Ball_bounced":'False', "Stroked_Player":'None', "Stroke_Type":'None'}, ignore_index=True)

        i += 1
        out.write(frame)
    out.release()

    return tennis_tracking

def find_bouning_point(blank_img, corrected_ball_position, bounced_model, point_detection_model):
    temp_img = blank_img.copy()
    ball_trajectory, (x_min, y_min), status = segment_ball_trajectory(temp_img, corrected_ball_position)
    
    if status == True: 

        h, w          = ball_trajectory.shape

        ball_trajectory = cv2.resize(ball_trajectory, (64, 64)) / 255.0
        ball_trajectory = ball_trajectory[np.newaxis,...,np.newaxis]
    
        with tf.device('cpu'):
            pred            = bounced_model.predict(ball_trajectory)

        
        if int(np.argmax(pred)) == 1:
            with tf.device('cpu'):
                pred           = point_detection_model.predict(ball_trajectory)[0, :, :, 0]

            heatmap        = (pred / pred.max() * 255).astype(np.uint8)

            draw_x, draw_y = keypoint_to_heatmap(heatmap)

            if draw_x != None:
                _, h_, w_,_         = ball_trajectory.shape
                draw_x, draw_y = draw_x/ w_, draw_y/h_


                draw_x, draw_y = int(draw_x * w + x_min) , int(draw_y*h + y_min)

                return draw_x, draw_y

    return None, None


def fit_polynomial(x, y, reg, poly_features):
    
    xdata          = np.array(x).reshape(-1,1)
    ydata          = np.array(y)
    
    X_poly         = poly_features.fit_transform(xdata)
    
    reg.fit(X_poly, ydata)

    y_pred         = reg.predict(X_poly)
    
    return y_pred, reg, poly_features

def l2_loss(y_pred, y_truth):
    return np.mean((y_pred - y_truth)**2)


def draw_output(q, draw, bounce_i, draw_bouncing, color):

    for i in range(q.shape[0]):
        if q[i, 0] is not None:
            draw_x = q[i, 0]
            draw_y = q[i, 1]
            bbox = (draw_x - 5, draw_y - 5, draw_x + 5, draw_y + 5)
            
            if bounce_i is not None and i == bounce_i:
                draw.ellipse(bbox, outline=color, fill=color)
                draw_bouncing.append(bbox)
            

    return draw, draw_bouncing

def draw_line(q, draw):
    ball_all = []
    for i in range(q.shape[0]):
        if q[i, 0] is not None:
            draw_x = q[i, 0]
            draw_y = q[i, 1]
            
            ball_all.append((draw_x, draw_y))

    draw.line(ball_all, fill ="green", width = 4)

    return draw



def mark_positions(frame, coordiantes, draw_bouncing, draw_stroke, bounce_index=None, stroke_index=None, mark_num=14, frame_num=None, ball_color='yellow'):
    
    # if frame number is not given, use the last positions found
    stroke_i, bounce_i = None, None

    if frame_num is not None:
        q = coordiantes[frame_num-mark_num+1:frame_num+1, :]

        for i in range(frame_num - mark_num + 1, frame_num + 1):
            if i in bounce_index:
                bounce_i = i - frame_num + mark_num - 1
                break

        for i in range(frame_num - mark_num + 1, frame_num + 1):
            if i in stroke_index:
                stroke_i = i - frame_num + mark_num - 1
                break
    
    else:
        q = coordiantes[-mark_num:, :]

    pil_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(pil_image)


    # Mark each position by a circle
    draw = ImageDraw.Draw(pil_image)


    for bounce_box in draw_bouncing:
        draw.ellipse(bounce_box, outline='red', fill='red')
    
    for stroke_box in draw_stroke:
        draw.ellipse(stroke_box, outline='blue', fill='blue')

    if bounce_i != None:
        draw, draw_bouncing = draw_output(q, draw, bounce_i, draw_bouncing, color = 'red')
    
    if stroke_i != None:
        draw, draw_stroke   = draw_output(q, draw, stroke_i, draw_stroke, color = 'blue')
    
        # Convert PIL image format back to opencv image format
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    del draw
    return frame

def ball_trajectory_and_bouncing(vid_path, raw_coordinates, blank_img, bounced_model, point_detection_model, queuelength=15):

    # In order to draw the trajectory of tennis, we need to save the coordinate of previous 7 frames
    q = queue.deque()
    for i in range(0, queuelength):
        q.appendleft({'data':None, 'frame_id':None})

    all_bouncing  = []
    frame_num     = 0

    video = cv2.VideoCapture(vid_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    print('fps : {}'.format(fps))
    output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc        = cv2.VideoWriter_fourcc(*'XVID')

    output_video = cv2.VideoWriter('./videos/ball_trajectory.mp4', fourcc, fps, (output_width, output_height))
  

    while True:

        ret, img = video.read()

        if ret:

            x, y = raw_coordinates[frame_num]

            if x == None:
                circles = None

            else:
                circles = [[x, y]]


            # check if there have any tennis be detected
            if circles is not None:
                # if only one tennis be detected
                if len(circles) == 1:

                    x = int(circles[0][0])
                    y = int(circles[0][1])

                    # push x,y to queue
                    q.appendleft({'data':[x, y], 'frame_id':frame_num})
                    # pop x,y from queue
                    q.pop()

                else:
                    # push None to queue
                    q.appendleft({'data':None, 'frame_id':frame_num})
                    # pop x,y from queue
                    q.pop()

            else:
                # push None to queue
                q.appendleft({'data':None, 'frame_id':frame_num})
                # pop x,y from queue
                q.pop()

            # previous_distance = None
            new_distance      = None
            # previous_state    = 1
            new_state         = 1
            # draw current frame prediction and previous 7 frames as yellow circle, total: 8 frames
            ball_all = []
            frames   = []
            
            if q[1]['data'] is not None:
                
                new_distance      = None
                new_state         = 1
                            
                start = 2
                end   = 5

                for j in range(start, end):
                    if  q[j]['data'] is not None:
                        new_distance = max(np.sqrt((q[1]['data'][0]-q[j]['data'][0])**2 + (q[1]['data'][1]-q[j]['data'][1])**2), 3.0)
                        break

                if new_distance !=None:    
                    new_state = j - 1

                else:
                    previous_distance = None
                    previous_state    = 1

                    for i in range(2, queuelength):
                        q[i]['data'] = None


                if (new_distance != None):
                    if ((new_distance / new_state) >= 50):
                        q[1]['data'] = None
                        #   print("False ball position")

                    else:
                        if (previous_distance is not None):
                        #   print("New_distance", (new_distance),"new_state", new_state, "previous distance", (previous_distance), "previous_state", previous_state)
                            if (new_distance / new_state) <= (2 * previous_distance / previous_state):

                                previous_distance = new_distance
                                previous_state    = new_state 

                            else:
                                if q[0]['data'] is not None:
                                    post_distance   = max(np.sqrt((q[0]['data'][0]-q[1]['data'][0])**2 + (q[0]['data'][1]-q[1]['data'][1])**2), 3.0)
                                    direct_distance = max(np.sqrt((q[0]['data'][0]-q[new_state + 1]['data'][0])**2 + (q[0]['data'][1]-q[new_state + 1]['data'][1])**2), 3.0)
                                    
                                    if (post_distance > 1.5 * (direct_distance / new_state)):
                                        q[1]['data'] = None
                                    #   print("False ball position")

                                    else:
                                        # print("Initialize")
                                        previous_distance = new_distance
                                        previous_state    = new_state 
                            
                        else:
                            # print("Initialize")
                            previous_distance = new_distance
                            previous_state    = new_state 

            for i in range(1, queuelength):
                if q[i]['data'] is not None:
                    draw_x = q[i]['data'][0]
                    draw_y = q[i]['data'][1]

                    frames.append(int(q[i]['frame_id']))
                    ball_all.append((int(draw_x), int(draw_y)))
            
            if len(ball_all)>4:
                x, y      = find_bouning_point(blank_img, ball_all, bounced_model, point_detection_model)
                    
                if x!=None:
                    
                    distance  = np.array([np.sqrt((x_f - x)**2 + (y_f - y)**2) for x_f, y_f in ball_all])
                    min_index = np.argmin(distance)
                    bouncing_point = frames[min_index]

                    all_bouncing.append(bouncing_point)

                PIL_image = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
                PIL_image = Image.fromarray(PIL_image)

                draw = ImageDraw.Draw(PIL_image)
                draw.line(ball_all, fill ="green", width = 4)

                img = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)

            output_video.write(img)
        

        else:
            break

        # if frame_num >= 100:
        #     break

        frame_num += 1
            # bouncing_points = [item for item in list(set(bouncing_points)) if bouncing_points.count(item)>2]

    output_video.release()
    video.release()

    return all_bouncing


# def find_all_bounding(ball_coordinate, blank_img, bounced_model, point_detection_model, mark_num=10):

#     bouncing_points = []
#     all_bouncing    = []
#     all_ball_corrdinate = [{'data':pos, 'frameid':ind} for ind, pos in enumerate(ball_coordinate)]

#     for frame_num in range(len(all_ball_corrdinate)):
       
        
#         q = all_ball_corrdinate[frame_num-mark_num+1:frame_num+1]
          
       
#         ball_all = []
#         frames   = []
        
#         for i in range(len(q)):
#             if q[i]['data'][0] is not None:
#                 draw_x = q[i]['data'][0]
#                 draw_y = q[i]['data'][1]

#                 frames.append(int(q[i]['frameid']))
#                 ball_all.append((int(draw_x), int(draw_y)))

#         if len(ball_all) > 4:
           
#             x, y      = find_bouning_point(blank_img, ball_all, bounced_model, point_detection_model)
            
#             if x!=None:
                
#                 distance  = np.array([np.sqrt((x_f - x)**2 + (y_f - y)**2) for x_f, y_f in ball_all])
#                 min_index = np.argmin(distance)
                
#                 bouncing_points.append(frames[min_index])
        

#     # Filter noises from real bouncing
#     bouncing_points = [item for item in list(set(bouncing_points)) if bouncing_points.count(item)>2]

#     return bouncing_points
    
def make_video(video_path, bouncing_points, stroke_points, ball_detector):
    video = cv2.VideoCapture(video_path)

    output_width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    output_video = cv2.VideoWriter('./output.mp4', fourcc, fps, (output_width, output_height))   

    ind = 0
   
    print(bouncing_points)
    draw_bouncing = []
    draw_stroke   = []

    while True:
        ret, img = video.read()
        if ret:
            
            img = mark_positions(img, ball_detector.xy_coordinates, draw_bouncing, draw_stroke, bounce_index=bouncing_points, stroke_index=stroke_points,mark_num=10, frame_num=ind, ball_color='green')       
            output_video.write(img)
            ind += 1

        else:
            break

    video.release()
    output_video.release()
    cv2.destroyAllWindows()


def draw_ball_position(frame, court_detector, xy, i):
    """
    Calculate the ball position of both players using the inverse transformation of the court and the x, y positions
    """
    inv_mats = court_detector.game_warp_matrix[i]
    coord = xy
    img = frame.copy()
    # Ball locations
    if coord is not None:
        p = np.array(coord,dtype='float64')
        ball_pos = np.array([p[0].item(), p[1].item()]).reshape((1, 1, 2))
        transformed = cv2.perspectiveTransform(ball_pos, inv_mats)[0][0].astype('int64')
        cv2.circle(frame, (transformed[0], transformed[1]), 35, (0,255,255), -1)
    else:
        pass
    return img 

def store_contact_points(tennis_tracking, bounce_index, stroke_index_p1, stroke_index_p2, stroke_action):

    for bounce_i in bounce_index:
        index = tennis_tracking[tennis_tracking['Frame']==bounce_i].index.to_numpy()[0]
        tennis_tracking.at[index, "Ball_bounced"] = 'True'
    
    for stroke in stroke_index_p1:
        action = stroke_action[stroke]['stroke']
        index  = tennis_tracking[tennis_tracking['Frame']==stroke].index.to_numpy()[0]
        tennis_tracking.at[index, "Stroked_Player"] = "P1"
        tennis_tracking.at[index, "Stroke_Type"] = action
       
    for stroke in stroke_index_p2:
        index = tennis_tracking[tennis_tracking['Frame']==stroke].index.to_numpy()[0]
        tennis_tracking.at[index, "Stroked_Player"] = "P2"
       
    return tennis_tracking
        
        




def video_process(video_path, show_video=False, include_video=True,
                  stickman=True, stickman_box=True, court=True,
                  output_file='output', output_folder='output',
                  smoothing=True, top_view=True):

    dtype = get_dtype()

    # initialize extractors
    # court_detector  = CourtDetector()
    # detection_model = DetectionModel(dtype=dtype)
    # pose_extractor  = PoseExtractor(person_num=1, box=stickman_box, dtype=dtype) if stickman else None
    # stroke_recognition = ActionRecognition('storke_classifier_weights.pth')
    ball_detector = BallDetector('./weights/model.3', out_channels=2)

    tennis_tracking = pd.DataFrame(columns=["Time", "Frame", "Player_1", "Player_2", "Ball_POS", "Ball_bounced", "Stroked_Player", "Stroke_Type"])


    # Bounce detection model
    img_size = (64, 64)
    bounced_model    = create_model(img_size, 0.000001, 2000)
    bounced_model.build(input_shape=(None, img_size[0], img_size[1], 1))
    bounced_model.load_weights('./bounce_model.h5')

    point_detection_model = tf.keras.models.load_model('./bouncing_point_detection.h5')

    # Load videos from videos path
    video = cv2.VideoCapture(video_path)

    # get videos properties
    fps, length, v_width, v_height = get_video_properties(video)

    blank_img   =  ((np.ones((v_height, v_width, 3)))*255).astype(np.uint8)

    # frame counter
    frame_i = 0

    # time counter
    total_time = 0

    # Loop over all frames in the videos
    while True:
        start_time = time.time()
        ret, frame = video.read()
        frame_i += 1

        if ret:
            # if frame_i == 1:
            #     court_detector.detect(frame)
            #     print(f'Court detection {"Success" if court_detector.success_flag else "Failed"}')
            #     print(f'Time to detect court :  {time.time() - start_time} seconds')
            #     start_time = time.time()

            # court_detector.track_court(frame)

            # # detect
            # detection_model.detect_player_1(frame.copy(), court_detector)
            # detection_model.detect_top_persons(frame, court_detector, frame_i)

            # # Create stick man figure (pose detection)
            # if stickman:
            #     pose_extractor.extract_pose(frame, detection_model.player_1_boxes)

            ball_detector.detect_ball(frame)

            total_time += (time.time() - start_time)
            print('Processing frame %d/%d  FPS %04f' % (frame_i, length, frame_i / total_time), '\r', end='')
            if not frame_i % 100:
                print('')

            # if frame_i > 100:
            #     break

        else:
            break

    # detection_model.find_player_2_box()

    corrds = []
    for i in ball_detector.xy_coordinates:
        if i[0]==None:
            corrds.append(None)

        else:
            corrds.append(i)


    # if top_view:
    # tennis_tracking = create_top_view(court_detector, detection_model, corrds, fps, tennis_tracking)

    # # Save landmarks in csv files
    # df = None
    # # # Save stickman data
    # if stickman:
    #     df = pose_extractor.save_to_csv(output_folder)

    # # smooth the output data for better results
    # df_smooth = None
    # if smoothing:
    #     smoother = Smooth()
    #     df_smooth = smoother.smooth(df)
    #     smoother.save_to_csv(output_folder)


    # player_1_strokes_indices, player_2_strokes_indices, bounces_indices, f2_x, f2_y = find_strokes_indices(
    #     detection_model.player_1_boxes,
    #     detection_model.player_2_boxes,
    #     ball_detector.xy_coordinates,
    #     df_smooth)

    bounces_indices     = ball_trajectory_and_bouncing(video_path, ball_detector.xy_coordinates, blank_img, bounced_model, point_detection_model)
    bouncing_points     = [item for item in list(set(bounces_indices)) if bounces_indices.count(item)>2]

    # stroke_indices      = list(player_1_strokes_indices) + list(player_2_strokes_indices)
    # temp_player_indices = stroke_indices + list(np.array(stroke_indices) - 2) + list(np.array(stroke_indices) - 1) + list(np.array(stroke_indices) + 2) + list(np.array(stroke_indices) + 1)

    # bounces_indices    = list(set(bounces_indices) - set(temp_player_indices))


    # '''ball_detector.bounces_indices = bounces_indices
    # ball_detector.coordinates = (f2_x, f2_y)'''
    # predictions = get_stroke_predictions(video_path, stroke_recognition,
                                        #  player_1_strokes_indices, detection_model.player_1_boxes)

    # tennis_tracking = store_contact_points(tennis_tracking, bounces_indices, player_1_strokes_indices, player_2_strokes_indices, predictions)
    make_video('./videos/ball_trajectory.mp4', bouncing_points, [], ball_detector)
    
    tennis_tracking.to_csv('statistics.csv', index = False)

    # print(predictions)

    # statistics = Statistics(court_detector, detection_model)
    # heatmap = statistics.get_player_position_heatmap()
    # statistics.display_heatmap(heatmap, court_detector.court_reference.court, title='Heatmap')
    # statistics.get_players_dists()

    # add_data_to_video(input_video=video_path, court_detector=court_detector, players_detector=detection_model,
    #                   ball_detector=ball_detector, strokes_predictions=predictions, skeleton_df=df_smooth,
    #                   statistics=statistics,
    #                   show_video=show_video, with_frame=1, output_folder=output_folder, output_file=output_file,
    #                   p1=player_1_strokes_indices, p2=player_2_strokes_indices, f_x=f2_x, f_y=f2_y)

    # # ball_detector.show_y_graph(detection_model.player_1_boxes, detection_model.player_2_boxes)


def main():
    s = time.time()
    os.makedirs('videos', exist_ok=True)
    video_process(video_path='./testing_data/custom_video_2.mkv', show_video=True, stickman=True, stickman_box=False, smoothing=True,
                  court=True, top_view=True)
    print(f'Total computation time : {time.time() - s} seconds')


if __name__ == "__main__":
    main()
