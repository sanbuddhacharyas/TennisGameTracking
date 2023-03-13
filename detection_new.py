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

from utils import get_video_properties, get_dtype, get_stickman_line_connection

from tennis_game_analysis import find_outcome_of_shot
import matplotlib.pyplot as plt

import pandas as pd
import queue

import imutils

from pickle import load
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
    video_length = 3
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


def find_strokes_indices(player_1_boxes, player_2_boxes, ball_positions, v_height, skeleton_df, court_detector, verbose=0):
    """
    Detect strokes frames using location of the ball and players
    """

    ball_positions      = np.array(ball_positions)

    with open('/home/predator/Desktop/UPWORK/Tennis_tracking/tennis-tracking/event_classifier.pkl', 'rb') as f:
        event_classifier = load(f)

    ball_x, ball_y = ball_positions[:, 0], ball_positions[:, 1]
    smooth_x = signal.savgol_filter(ball_x, 3, 2)
    smooth_y = signal.savgol_filter(ball_y, 3, 2)

    # Ball position interpolation
    x = np.arange(0, len(smooth_y))
    indices = [i for i, val in enumerate(smooth_y) if np.isnan(val)]
    x  = np.delete(x, indices)
    y1 = np.delete(smooth_y, indices)
    y2 = np.delete(smooth_x, indices)
    ball_f2_y = interp1d(x, y1, kind='cubic', fill_value="extrapolate")
    ball_f2_x = interp1d(x, y2, kind='cubic', fill_value="extrapolate")
    xnew      = np.linspace(0, len(ball_y), num=len(ball_y), endpoint=True)


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

    ball_filter_y = ball_f2_y(xnew)
    ball_filter_x = ball_f2_x(xnew)

    # Find all peaks of the ball y index
    p, _     = find_peaks(np.array(ball_filter_y), width=1)
    neg_p, _ = find_peaks(max(ball_filter_y) - np.array(ball_filter_y), width=6)

    all_peaks    = sorted(list(p) + list(neg_p))
    left_right_margin = 20 

    bounces_ind  = []
    peaks        = []
    neg_peaks    = []

    ball_traj_y_nor = np.array([i/v_height for i in ball_filter_y])

    plt.plot(ball_traj_y_nor)
    plt.savefig('all_data.jpg')
    plt.clf()

    for i in all_peaks:
        data = ball_traj_y_nor[i - left_right_margin : i + left_right_margin]

        plt.plot(data)
        plt.savefig(f'{i}.jpg')
        plt.clf()

        X_t = np.array(data)[np.newaxis,...]

        print(X_t.shape[1])
     
        if X_t.shape[1]==40:
            p    = event_classifier.predict(X_t)[0]
            prob = max(event_classifier.predict_proba(X_t)[0])

            print(p, prob, i)

            if (p==1) and (prob>0.8):
                bounces_ind.append(i)

            elif p==3 and prob>0.8:
                bounces_ind.append(i)

            elif p==2 and prob>0.8:
                peaks.append(i)

            elif p==4 and prob>0.9:
                neg_peaks.append(i)
        
    
    # Get bottom player wrists positions
    left_wrist_index = 9
    right_wrist_index = 10
    # skeleton_df = skeleton_df.fillna(-1)
    # left_wrist_pos  = skeleton_df.iloc[:, [left_wrist_index, left_wrist_index + 15]].values
    # right_wrist_pos = skeleton_df.iloc[:, [right_wrist_index, right_wrist_index + 15]].values

    dists = []

    # Calculate dist between ball and bottom player
    for i, player_box in enumerate(player_1_boxes):
        if player_box[0] is not None:
            player_center = center_of_box(player_box)
            Ball_POS = np.array([ball_f2_x(i), ball_f2_y(i)])
            box_dist = np.linalg.norm(player_center - Ball_POS)
            right_wrist_dist, left_wrist_dist = np.inf, np.inf

           
            try:
                # if right_wrist_pos[i, 0] > 0:
                #     right_wrist_dist = np.linalg.norm(right_wrist_pos[i, :] - Ball_POS)
                # if left_wrist_pos[i, 0] > 0:
                #     left_wrist_dist = np.linalg.norm(left_wrist_pos[i, :] - Ball_POS)
                dists.append(box_dist)
            
            except:
                dists.append(None)

        else:
            dists.append(None)

    dists = np.array(dists)

    dists2 = []
    
    # Calculate dist between ball and top player
    for i in range(len(player_2_centers)):
        Ball_POS = np.array([ball_f2_x(i), ball_f2_y(i)])
        box_center = np.array([player_2_f_x(i), player_2_f_y(i)])
        box_dist = np.linalg.norm(box_center - Ball_POS)
        dists2.append(box_dist)
    dists2 = np.array(dists2)

    strokes_1_indices = []

    # Find stroke for bottom player by thresholding the dists
    for peak in peaks:
        player_box_height = max(player_1_boxes[peak][3] - player_1_boxes[peak][1], 130)

        if dists[peak] != None:
            if dists[peak] < (player_box_height * 4 / 5):
                strokes_1_indices.append(peak)

    strokes_2_indices = []
    netting = []

    # Find stroke for top player by thresholding the dists
    for peak in neg_peaks:
        if dists2[peak] != None:
            if dists2[peak] < 100:
                strokes_2_indices.append(peak)

            else:
                inv_mats    = court_detector.game_warp_matrix[peak]
                p           = np.array([ball_x[peak], ball_y[peak]],dtype='float64')
                Ball_POS    = np.array([p[0].item(), p[1].item()]).reshape((1, 1, 2))
                transformed = cv2.perspectiveTransform(Ball_POS, inv_mats)[0][0].astype('int64')
                
                if (transformed[0]>0) and (transformed[1]>0):
                    netting.append(peak)

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

    bal_pos_filter = [[x, y] for x, y in zip(ball_filter_x, ball_filter_y)]

    return strokes_1_indices, strokes_2_indices, bounces_ind, player_2_f_x, player_2_f_y, netting, bal_pos_filter


def mark_player_box(frame, boxes, frame_num):
    box = boxes[frame_num]
    if box[0] is not None:
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [255, 0, 0], 2)
    return frame


def mark_skeleton(skeleton_df, img, img_no_frame, frame_number):
    """
    Mark the skeleton of the bottom player on the frame
    """
    # landmarks colors
    circle_color, line_color = (0, 0, 255), (255, 0, 0)
    stickman_pairs = get_stickman_line_connection()

    skeleton_df = skeleton_df.fillna(-1)

    try:
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

    except:
        pass

    return img, img_no_frame


def merge(frame, mini_img):
    frame_w = frame.shape[1]

    width   = frame_w // 5
    resized = imutils.resize(mini_img, width=width)

    img_w = resized.shape[1]
    img_h = resized.shape[0]

    w = frame_w - img_w

    frame[:img_h, w:] = resized

    return frame 

def add_data_to_video(input_video, court_detector, players_detector, ball_detector, strokes_predictions_1, strokes_predictions_2, skeleton_df,
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

    minimap = cv2.VideoCapture('./minimap.mp4')

    # videos properties
    fps, length, width, height = get_video_properties(cap)
    final_width = width * 2 if with_frame == 2 else width

    fps, length, width, height = get_video_properties(cap)

    # Video writer
    out = cv2.VideoWriter(os.path.join(output_folder, output_file + '.mp4'),
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (final_width, height))

    # initialize frame counters
    frame_number = 0
    orig_frame = 0
    while True:
        orig_frame += 1
        print('Creating new videos frame %d/%d  ' % (orig_frame, length), '\r', end='')
        if not orig_frame % 100:
            print('')
        ret, img   = cap.read()
        _, min_img = minimap.read()

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
        # img = ball_detector.mark_positions(img, frame_num=frame_number)
        # img_no_frame = ball_detector.mark_positions(img_no_frame, frame_num=frame_number, ball_color='black')

        # add pose stickman
        # if skeleton_df is not None:
        #     img, img_no_frame = mark_skeleton(skeleton_df, img, img_no_frame, frame_number)

        # Add stroke prediction
        for i in range(-10, 10):
            if frame_number + i in strokes_predictions_1.keys():
                '''cv2.putText(img, 'STROKE HIT', (200, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255) if i != 0 else (255, 0, 0), 3)'''

                probs, stroke = strokes_predictions_1[frame_number + i]['probs'], strokes_predictions_1[frame_number + i][
                    'stroke']
            
                cv2.putText(img, f'Stroke : {stroke}',
                            (int(player1_boxes[frame_number][0]) - 10, int(player1_boxes[frame_number][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                break

            if frame_number + i in strokes_predictions_2.keys():
                '''cv2.putText(img, 'STROKE HIT', (200, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255) if i != 0 else (255, 0, 0), 3)'''

                probs, stroke = strokes_predictions_2[frame_number + i]['probs'], strokes_predictions_2[frame_number + i][
                    'stroke']
            
                if player2_boxes[frame_number][0]!=None:
                    cv2.putText(img, f'Stroke : {stroke}',
                                (int(player2_boxes[frame_number][0]) - 5, int(player2_boxes[frame_number][1]) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                break
        # # Add stroke detected
        # for i in range(-5, 10):
        #     '''if frame_number + i in p1:
        #         cv2.putText(img, 'Stroke detected', (int(player1_boxes[frame_number][0]) - 10, int(player1_boxes[frame_number][1]) - 10),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255) if i != 0 else (255, 0, 0), 2)'''

        #     if frame_number + i in p2:
        #         cv2.putText(img, 'Stroke detected',
        #                     (int(f_x(frame_number)) - 30, int(f_y(frame_number)) - 50),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255) if i != 0 else (255, 0, 0), 2)

        cv2.putText(img, 'Distance: {:.2f} m'.format(player1_dists[frame_number] / 100),
                    (50, 500),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(img, 'Distance: {:.2f} m'.format(player2_dists[frame_number] / 100),
                    (100, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


        # # display frame
        # if show_video:
        #     cv2.imshow('Output', img)
        #     if cv2.waitKey(1) & 0xff == 27:
        #         cv2.destroyAllWindows()

        # save output videos
        if with_frame == 0:
            final_frame = img_no_frame

        elif with_frame == 1:
            final_frame = img
        else:
            final_frame = np.concatenate([img, img_no_frame], 1)

        final_frame = merge(final_frame, min_img)
        out.write(final_frame)
        frame_number += 1
    print('Creating new video frames %d/%d  ' % (length, length), '\n', end='')
    print(f'New videos created, file name - {output_file}.avi')
    cap.release()
    out.release()
    cv2.destroyAllWindows()


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
    print("MAX", MAX)
    if MAX == [0]:
      pass
    else:
      try:
        print("Tuple", tuple(MAX))
        coords[coords.index(tuple(MAX))] = None
      except ValueError:

        try:
            coords[coords.index(MAX)] = None
        except:
            pass

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

def fill_tennis_status(court_detector, detection_model, coords, fps, tennis_tracking):
    
    # players location on court
    smoothed_1, smoothed_2 = detection_model.calculate_feet_positions(court_detector)
    i = 0 

    for feet_pos_1, feet_pos_2 in zip(smoothed_1, smoothed_2):
        inv_mats = court_detector.game_warp_matrix[i]

        p           = np.array(coords[i],dtype='float64')
        Ball_POS    = np.array([p[0].item(), p[1].item()]).reshape((1, 1, 2))
        ball_transformed = cv2.perspectiveTransform(Ball_POS, inv_mats)[0][0].astype('int64')


        row = {"Time": (i / fps), "Frame":i, "Player_Near_End_Pos":(int(feet_pos_1[0]), int(feet_pos_1[1])), \
            "Player_Far_End_Pos":(int(feet_pos_2[0]), int(feet_pos_2[1])), "Ball_POS":ball_transformed, "Ball_bounced":'False', "Stroke_by":'None', "Stroke_Type":'None', "Ball_Bounce_Outcome":'None', 'Ball_Bounce_Pos':'None', 'Ball_predict_point': 'None'}
        
        tennis_tracking = tennis_tracking.append(row, ignore_index=True)  
        i += 1

    return tennis_tracking
   
def create_top_view(court_detector, detection_model, fps, tennis_tracking, volleyed):
    """
    Creates top view video of the gameplay
    """

    court = court_detector.court_reference.court.copy()
    print(court, court.shape)
    court = cv2.line(court, *court_detector.court_reference.net, 255, 5)
   
    v_width, v_height = court.shape[::-1]
    court  = cv2.cvtColor(court, cv2.COLOR_GRAY2BGR)
    out    = cv2.VideoWriter('./minimap.mp4',cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, (v_width, v_height))

    
    print("Court Shape", court.shape)
    players_trajectory  = []
    ball_bounced_points = []  


    arrow_points = None

    for _, row in tennis_tracking.iterrows():

        frame = court.copy()

        # Draw the players trajectory
        for player_trajectory in players_trajectory:
            frame = cv2.circle(frame, player_trajectory, 10, (255, 0, 0), -1)

        # Draw ball bouncing point
        for (ball_bouncing, color) in ball_bounced_points:
            frame = cv2.circle(frame, ball_bouncing, 20, color, -1)

        

        frame = cv2.circle(frame, row['Player_Near_End_Pos'], 35, (255, 0, 0), -1)

        if row['Player_Far_End_Pos'][0] is not None:
            frame = cv2.circle(frame, row['Player_Far_End_Pos'], 35, (255, 0, 0), -1)


        players_trajectory.append(row['Player_Near_End_Pos'])
        players_trajectory.append(row['Player_Far_End_Pos'])

        if row['Ball_Bounce_Outcome'] == 'volleyed':
            if row['Stroke_by'] == 'P1':
                ball_bouncing = tuple(row['Player_Near_End_Pos'])

            else:
                ball_bouncing = tuple(row['Player_Far_End_Pos'])

            color = (255, 255, 18)
            frame = cv2.circle(frame, ball_bouncing, 20, color, -1)
            ball_bounced_points.append([ball_bouncing, color])

        if row['Stroke_by'] == 'P1':
            color = (255, 0, 0)
            start = row['Player_Near_End_Pos']
            end   = row['Ball_predict_point']

        
            if (str(start) != 'None') and (str(end) != 'None'):
                frame = cv2.arrowedLine(img = frame, pt1 = start, pt2=end, color = (0, 255, 0), thickness=8)
                arrow_points = [start, end]

        if row['Stroke_by'] == 'P2':
            color = (255, 0, 0)
            start = row['Player_Far_End_Pos']
            end   = row['Ball_predict_point']

            if (str(start) != 'None') and (str(end) != 'None'):
                frame = cv2.arrowedLine(img = frame, pt1 = start, pt2=end, color= (0, 255, 0), thickness=8)
                arrow_points = [start, end]

        if arrow_points != None:
            start, end = arrow_points
            frame = cv2.arrowedLine(img = frame, pt1 = start, pt2=end, color= (0, 255, 0), thickness=8)


        if row['Ball_bounced']:

            ball_bouncing = tuple(row['Ball_POS'])
    
            if row['Ball_Bounce_Outcome']=='In':
                color = (45, 255, 255) 
                frame = cv2.circle(frame, ball_bouncing, 20, color, -1)
                ball_bounced_points.append([ball_bouncing, color])
                arrow_points = None

            elif row['Ball_Bounce_Outcome']=='Out':
                color = (0, 0, 255)
                frame = cv2.circle(frame, ball_bouncing, 20, color, -1)
                ball_bounced_points.append([ball_bouncing, color])
                arrow_points = None


        out.write(frame)

    out.release()


def interpolate_ball_trajectory(coords):
    # Remove Outliers 
    x, y = diff_xy(coords)
    remove_outliers(x, y, coords)

    # Interpolation
    coords = interpolation(coords)
    coords = np.array([[i[0], i[1]] for i in coords])

    return coords

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


def fit_polynomial_trajectory(ball_possition, reg, poly_features):

    temp_x         = []
    temp_y         = []

    prev_temp_x    = []
    prev_temp_y    = []
    prev_y_pred    = []

    prev_curve_x   = []
    prev_y_curv    = []

    
    for (x, y) in ball_possition:
        
        print(x, y)
        prev_flag = False

        if len(temp_x) < 3:
            temp_x.append(x)
            temp_y.append(y)

        else:
            temp_x.append(x)
            temp_y.append(y)

            y_pred, reg, poly_features = fit_polynomial(temp_x, temp_y, reg, poly_features)

            y_truth        = np.array(temp_y)
            error          = l2_loss(y_pred, y_truth)

            if error > 20:
                if len(prev_temp_x) >= 1:
                    temp_x     = [prev_temp_x[-1], x]
                    temp_y     = [prev_temp_y[-1], y]

                else:
                    temp_x     = [x]
                    temp_y     = [y]

                prev_flag = True

            else:
                prev_temp_x = temp_x.copy()
                prev_temp_y = temp_y.copy()
                prev_y_pred = list(y_pred).copy()

        if (prev_curve_x != []) and not(prev_flag):

            if len(prev_temp_x)==4:
                prev_curve_x      = prev_curve_x + prev_temp_x[1:]
                # prev_curve_y      = prev_curve_y + prev_temp_y[1:]
                prev_y_curv       = prev_y_curv  + prev_y_pred[1:]

            else:
                prev_curve_x      = prev_curve_x + [prev_temp_x[-1]]
                # prev_curve_y      = prev_curve_y + [prev_temp_y[-1]]
                prev_y_curv       = prev_y_curv  + [prev_y_pred[-1]]


        else:
            prev_curve_x      = prev_temp_x.copy()
            # prev_curve_y      = prev_temp_y.copy()
            prev_y_curv       = prev_y_pred.copy()  
    
    print(prev_curve_x, prev_y_curv)
    return [(int(x), int(y)) for x, y in zip(prev_curve_x, prev_y_curv)]


def l2_loss(y_pred, y_truth):
    return np.mean((y_pred - y_truth)**2)


def draw_output(draw, draw_points, draw_bouncing, color):
    
    draw_x = draw_points[0]
    draw_y = draw_points[1]
    bbox   = (draw_x - 5, draw_y - 5, draw_x + 5, draw_y + 5)
            
    draw.ellipse(bbox, outline=color, fill=color)
    draw_bouncing.append([bbox, color])
        

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



def mark_positions(frame, coordiantes, draw_bouncing, draw_stroke, bounce_index=None, stroke_index=None, netting=None, volleyed=None, mark_num=14, frame_num=None, is_action_true=None, ball_color='yellow'):

    pil_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(pil_image)

    draw = ImageDraw.Draw(pil_image)

    color = 'yellow'
    if frame_num in bounce_index:
        draw_point = coordiantes[frame_num]

        if is_action_true=='In':
            color = 'yellow'

        else:
            color = 'red'

        if draw_point[0] == None:
            if coordiantes[frame_num + 1][0] != None:
                draw_point = coordiantes[frame_num + 1]


            elif coordiantes[frame_num - 1][0] != None:
                draw_point = coordiantes[frame_num - 1]

        draw.text((draw_point[0]+5, draw_point[1]+5), str(is_action_true), fill=color)
        draw, draw_bouncing       = draw_output(draw, draw_point, draw_bouncing , color = color)


    if frame_num in netting:
        draw_point = coordiantes[frame_num]
        draw.text((draw_point[0]+5, draw_point[1]+5), "Netting", fill=color)
        draw, draw_stroke       = draw_output(draw, draw_point , draw_stroke, color = 'red')       

        

    if frame_num in stroke_index:
        draw_point = coordiantes[frame_num]

        if draw_point[0] == None:
            if coordiantes[frame_num + 1][0] != None:
                print("frame+1", draw_point)
                draw_point = coordiantes[frame_num + 1]

            elif coordiantes[frame_num - 1][0] != None:
                draw_point = coordiantes[frame_num - 1]
                print("frame-1", draw_point)

        if frame_num in volleyed:
            draw, draw_stroke       = draw_output(draw, draw_point , draw_stroke, color = 'orange')

        else:
            draw, draw_stroke       = draw_output(draw, draw_point , draw_stroke, color = 'blue')


    for bounce_box, color in draw_bouncing:
        draw.ellipse(bounce_box, outline=color, fill=color)
    
    for stroke_box, color in draw_stroke:
        draw.ellipse(stroke_box, outline=color, fill=color)

        # Convert PIL image format back to opencv image format
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    del draw
    return frame

def ball_trajectory_and_bouncing(vid_path, raw_coordinates, blank_img, bounced_model, point_detection_model, queuelength=15):

    # In order to draw the trajectory of tennis, we need to save the coordinate of previous 7 frames
    q = queue.deque()
    for i in range(0, queuelength):
        q.appendleft({'data':None, 'frame_id':None})

    # poly_features  = PolynomialFeatures(degree=4, include_bias=False)
    # reg            = LinearRegression()

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
                # ball_all = fit_polynomial_trajectory(ball_all, reg, poly_features)
                x, y      = find_bouning_point(blank_img, ball_all, bounced_model, point_detection_model)
                    
                if x!=None:
                    
                    distance  = np.array([np.sqrt((x_f - x)**2 + (y_f - y)**2) for x_f, y_f in ball_all])
                    min_index = np.argmin(distance)
                    bouncing_point = frames[min_index]

                    all_bouncing.append(bouncing_point)

                PIL_image = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
                PIL_image = Image.fromarray(PIL_image)

                draw = ImageDraw.Draw(PIL_image)
                

                # for size, draw_points in enumerate(ball_all):

                #     size   = size // 2
                #     draw_x = draw_points[0]
                #     draw_y = draw_points[1]
                #     bbox   = (draw_x - 7 + size, draw_y - 7 + size, draw_x + 7 - size, draw_y + 7 - size)
                            
                #     draw.ellipse(bbox, outline='yellow')
                draw.line(ball_all, fill ="yellow", width = 4)

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

    
def make_video(video_path, bouncing_points, stroke_points, netting, volleyed, coords, tennis_tracking):
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

    all_actions     = []
    hitting_player  = {}
    hitting_player['Stroke_by']    = None
    hitting_player['Stroke_Type']  = None
    hitting_player['Frame']        = None
    action_series = ''

    is_action_true  = None
    last_action     = 'Serve'

    hit_flag     = False
    first_shot   = False

    while True:
        ret, img = video.read()
        if ret:
            
            row = tennis_tracking.iloc[ind]
            first_shot, hitting_player, all_actions, hit_flag, last_action, is_action_true, action_series, img, volleyed = find_outcome_of_shot(row, first_shot, hitting_player, all_actions, hit_flag, last_action, is_action_true, action_series, img, ind, tennis_tracking, volleyed)
            img = mark_positions(img, coords, draw_bouncing, draw_stroke, bounce_index=bouncing_points, stroke_index=stroke_points, netting=netting, volleyed=volleyed, mark_num=10, frame_num=ind, is_action_true=is_action_true, ball_color='green')       
            output_video.write(img)
            ind += 1

        else:
            break

    video.release()
    output_video.release()
    cv2.destroyAllWindows()

    return volleyed


def draw_ball_position(frame, court_detector, coord, i):
    """
    Calculate the ball position of both players using the inverse transformation of the court and the x, y positions
    """
    inv_mats = court_detector.game_warp_matrix[i]
    
    # Ball locations
    if coord is not None:
        p           = np.array(coord,dtype='float64')
        Ball_POS    = np.array([p[0].item(), p[1].item()]).reshape((1, 1, 2))
        transformed = cv2.perspectiveTransform(Ball_POS, inv_mats)[0][0].astype('int64')
        # cv2.circle(frame, (transformed[0], transformed[1]), 30, (45, 255, 255), -1)

    else:
        pass

    return frame, (transformed[0], transformed[1])

def store_contact_points(tennis_tracking, bounce_index, stroke_index_p1, stroke_index_p2, stroke_action_1, stroke_action_2, netting):

    for bounce_i in bounce_index:

        index = tennis_tracking[tennis_tracking['Frame']==bounce_i].index.to_numpy()[0]
        tennis_tracking.at[index, "Ball_bounced"] = 'True'


    for net_i in netting:
        index = tennis_tracking[tennis_tracking['Frame']==net_i].index.to_numpy()[0]
        tennis_tracking.at[index, "Ball_Bounce_Outcome"] = 'Netting'

    
    for stroke in stroke_index_p1:
        action = stroke_action_1[stroke]['stroke']
        index  = tennis_tracking[tennis_tracking['Frame']==stroke].index.to_numpy()[0]
        tennis_tracking.at[index, "Stroke_by"] = "P1"
        tennis_tracking.at[index, "Stroke_Type"] = action
       
    for stroke in stroke_index_p2:
        action = stroke_action_2[stroke]['stroke']
        index  = tennis_tracking[tennis_tracking['Frame']==stroke].index.to_numpy()[0]
        tennis_tracking.at[index, "Stroke_by"] = "P2"
        tennis_tracking.at[index, "Stroke_Type"] = action
       
    return tennis_tracking

def bouncing_filter(bouncing_indices, invmats, ball_coordinates, threshold= 3):
    
    bouncing_indices     = sorted(bouncing_indices)
    valid_bounce         = []

    for ind in bouncing_indices:

        inv_mats    = invmats[ind]
        p           = np.array(ball_coordinates[ind],dtype='float64')
        Ball_POS    = np.array([p[0].item(), p[1].item()]).reshape((1, 1, 2))
        transformed = cv2.perspectiveTransform(Ball_POS, inv_mats)[0][0].astype('int64')

        if (transformed[0]>0) and (transformed[1] >0):
            valid_bounce.append(ind)

  
    # new_bounce  = []
    # prev_score = 0
    # prev_frame = 0

    # for (frame_num, score) in valid_bounce:
    #     if score >= 2:
    #         # print("score>2", frame_num, score)
    #         if len(new_bounce) == 0:
    #             new_bounce.append(frame_num)
    #             prev_score = score
    #             prev_frame = frame_num


    #         else:
    #             if abs(prev_frame - frame_num) <= threshold:
    #                 # print("bello threshold", frame_num, score,prev_frame, prev_score)
    #                 if score >= prev_score:
    #                     # print("below threshold", frame_num, score, prev_frame, prev_score)
    #                     new_bounce[-1] = frame_num
    #                     prev_score = score
    #                     prev_frame = frame_num


    #             else:
    #                 new_bounce.append(frame_num)
    #                 prev_score = score
    #                 prev_frame = frame_num

    return valid_bounce


        
        

def find_manual_homographic_transformation(frame, court_detector, num_frames):
    global intersections, img

    intersections     = []
    img               = frame.copy()
    court_detector.court_warp_matrix = []
    court_detector.game_warp_matrix  = []

    def draw_circle(event,x,y,flags,param):
        global intersections, img
        
        if event == cv2.EVENT_LBUTTONDOWN:
            intersections.append([x, y])
            img = cv2.circle(img,(x,y),4,(255,0,0),-1)
            cv2.imshow('court_tracker', img)
            
    cv2.namedWindow('court_tracker')
    cv2.setMouseCallback('court_tracker',draw_circle)

    cv2.imshow('court_tracker', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    configuration = [[286, 561], [1379, 561], [286, 2935], [1379, 2935]]
    matrix, _ = cv2.findHomography(np.float32(configuration), np.float32(intersections), method=0)
    inv_matrix = cv2.invert(matrix)[1]

    for i in range(num_frames):
        court_detector.court_warp_matrix.append(matrix)
        court_detector.game_warp_matrix.append(inv_matrix)

    return court_detector
