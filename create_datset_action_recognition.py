
import cv2
import pandas as pd
import numpy as np
import imageio
import os

from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

import time
import matplotlib.pyplot as plt

from glob import glob
from detect_players import detect_player_ball
from tqdm import tqdm


from detection import DetectionModel
from ultralytics import YOLO
from utils_ import  get_video_properties, get_dtype
from detection_new import ball_trajectory_filter, find_min_dist
from detection import DetectionModel, center_of_box


def find_strokes_indices(player_1_boxes, player_2_boxes, ball_filtered, v_width, v_height):
    """
    Detect strokes frames using location of the ball and players
    """

    (ball_filter_x, ball_filter_y, ball_f2_x, ball_f2_y) = ball_filtered

    # Player 2 position interpolation
    player_2_centers = np.array([center_of_box(box) for box in player_2_boxes])
    print("player_2_boxes", len(player_2_boxes))
    print("player_2_centers_shape",player_2_centers.shape)
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


    # Find all peaks of the ball y index
    peaks, _     = find_peaks(np.array(ball_filter_y))
    neg_peaks, _ = find_peaks(max(ball_filter_y) - np.array(ball_filter_y), width=6)


    dists = []

    # Calculate dist between ball and bottom player
    for i, player_box in enumerate(player_1_boxes):
        if player_box[0] is not None:
         
            Ball_POS   = np.array([ball_f2_x(i), ball_f2_y(i)])
            player_box = np.array([player_box[0]/v_width, player_box[1]/v_height, player_box[2]/v_width, player_box[3]/v_height])
            box_dist = find_min_dist(Ball_POS, player_box)
           
            try:
                dists.append(box_dist)
            
            except:
                dists.append(None)

        else:
            dists.append(None)

    dists = np.array(dists)

    dists2 = []
    
    # Calculate dist between ball and top player
    for i, player_box in enumerate(player_2_boxes):
        if player_box[0] is not None:
            Ball_POS   = np.array([ball_f2_x(i), ball_f2_y(i)])
            player_box = np.array([player_box[0]/v_width, player_box[1]/v_height, player_box[2]/v_width, player_box[3]/v_height])
            box_dist   = find_min_dist(Ball_POS, player_box)
           
            try:
                dists2.append(box_dist)
            
            except:
                dists2.append(None)

        else:
            dists2.append(None)

    dists2 = np.array(dists2)

    strokes_1_indices = []

    # Find stroke for bottom player by thresholding the dists
    for peak in peaks:
        if player_1_boxes[peak][0]!=None:
            player_box_height = max(player_1_boxes[peak][3]/v_height - player_1_boxes[peak][1]/v_height, 0.2)

            if dists[peak] != None:
                if dists[peak] < (player_box_height * 4 / 5):
                    strokes_1_indices.append(peak)

    strokes_2_indices = []

    # Find stroke for top player by thresholding the dists
    for peak in neg_peaks:
        if dists2[peak] != None:
            if dists2[peak] < 0.15:
                strokes_2_indices.append(peak)
          
                # if (transformed[0]>0) and (transformed[1]>0):
                #     netting.append(peak)

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

    

    return strokes_1_indices, strokes_2_indices

def make_gif(video, frame_id, frame_count, tennis_tracking, player_mode, save_name):
    
    start    = max(0, (frame_id - 20))
    range_   = (frame_id + 15 - start)

    print("start", start)
    print("end", range_)
    video.set(cv2.CAP_PROP_POS_FRAMES, start)

    game_play = []
    for i in range(range_):
        res, frame = video.read()
        if res:
           
            bbox  = tennis_tracking[tennis_tracking['Frame']==start+i][player_mode].to_numpy()

            if len(bbox)!=0:
                if bbox[0][0] != None:
                    player = crop_players(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), bbox[0])

                    try:
                        player = cv2.resize(player, (300, 400))
                        game_play.append(player)

                    except:
                        player = game_play[-1]
                        game_play.append(player)


                else:
                    if len(game_play)>0:
                        player = game_play[-1]
                        game_play.append(player)

    imageio.mimsave(f'{save_name}.gif', game_play, fps=20)

def crop_players(frame, bbox, margin=20):
    return frame[bbox[1]-margin:bbox[3]+margin, bbox[0]-margin:bbox[2]+margin, :]

def find_csv_file_game(video_path, save_location):
    dtype = get_dtype()

    print("video_path", video_path)
    video = cv2.VideoCapture(video_path)

    detection_model   = DetectionModel(dtype=dtype)
    player_ball_model = YOLO('./weights/best.pt') 

    tennis_tracking       = pd.DataFrame(columns=["vid_name", "Frame", "Player_1_bbox", "Player_2_bbox", "Stroke_by", "Stroke_Type","gif_path"])

    # get videos properties
    fps, length, v_width, v_height = get_video_properties(video)

    # frame counter
    frame_i = 0

    # time counter
    total_time = 0

    output       = False

    # # Bounce detection model
    # img_size = (64, 64)
    # bounced_model    = create_model(img_size, 0.000001, 2000)
    # bounced_model.build(input_shape=(None, img_size[0], img_size[1], 1))
    # bounced_model.load_weights('./bounce_model.h5')

    yolo_ball_pos = []
    

    while True:
        start_time = time.time()
        ret, frame = video.read()

        if ret:
            # detect
            player_1, player_2, ball_pos = detect_player_ball(player_ball_model, frame.copy())
            detection_model.player_1_boxes.append(player_1)
            detection_model.player_2_boxes.append(player_2)

            total_time += (time.time() - start_time)
            print('Processing frame %d/%d  FPS %04f' % (frame_i, length, frame_i / total_time), '\r', end='')
            if not frame_i % 100:
                print('')

        else:
            break

        frame_i += 1

    
    
    print("Player detection completed")
    
    ball_trajectory_path = video_path.replace('.mp4', '.npy').replace('.mkv', '.npy')
    xy_coordinates       = np.load(ball_trajectory_path, allow_pickle=True)

    plt.plot(np.array(xy_coordinates)[:,1])
    plt.savefig('./ball_plots/yolo_ball_pos.jpg')
    plt.cla()



    ball_filtered, coords = ball_trajectory_filter(xy_coordinates)
    print('coords', len(coords))
  
    player_1_strokes_indices, player_2_strokes_indices = find_strokes_indices(
        detection_model.player_1_boxes,
        detection_model.player_2_boxes,
        ball_filtered,
        v_width,
        v_height,
        )
    
    print(player_1_strokes_indices, player_2_strokes_indices)
    vid_name = video_path.split('/')[-1].split('.')[0]
    
    frame_i = 0

    print("video_length", length)
    for player_1, player_2 in zip(detection_model.player_1_boxes, detection_model.player_2_boxes):
        stroke_by = 'None'
        gif_path  = 'None'

        if frame_i in player_1_strokes_indices:
            stroke_by = "P1"
            gif_path  = f"{save_location}/{video_path.split('/')[-1].split('.')[0]}_{frame_i}_{stroke_by}"
        

        if frame_i in player_2_strokes_indices:
            stroke_by = "P2"      
            gif_path  = f"{save_location}/{video_path.split('/')[-1].split('.')[0]}_{frame_i}_{stroke_by}"

        row = {"vid_name":vid_name, "Frame":frame_i, "Player_1_bbox":player_1, "Player_2_bbox":player_2, "Stroke_by":stroke_by, "Stroke_Type":'None','gif_path':gif_path}
        tennis_tracking = tennis_tracking.append(row, ignore_index=True)

        frame_i += 1

    csv_loc = video_path.replace('.mp4', '.pkl').replace('.mkv', '.pkl')
    tennis_tracking.to_pickle(csv_loc)

    for _, row in tennis_tracking.iterrows():

        if row['Stroke_by']=="P1":
            save_name = row['gif_path']
            frame_i   = row['Frame']
            make_gif(video, frame_i, length, tennis_tracking, "Player_1_bbox", save_name)

        if row['Stroke_by']=="P2":
            save_name = row['gif_path']
            frame_i   = row['Frame']
            make_gif(video, frame_i, length, tennis_tracking, "Player_2_bbox", save_name)
        
    video.release()


if __name__=='__main__':
    vid_path = '/home/predator/Desktop/UPWORK/Tennis_tracking/tennis-tracking/event_detection_dataset'
    all_vid_paths = sorted(glob(f'{vid_path}/*.mp4') + glob(f'{vid_path}/*.mkv'))
    os.makedirs('action_recognition_player', exist_ok=True)
    for vid_path in tqdm(all_vid_paths[10:]):

        # try:
        find_csv_file_game(vid_path)

        # except:
        #     pass

    
   