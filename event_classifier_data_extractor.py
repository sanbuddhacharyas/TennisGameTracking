import os
import time

from PIL import Image, ImageDraw
import tensorflow as tf

import cv2
import numpy as np

from glob import glob



from ball_detection import BallDetector
from tqdm import tqdm
from utils import get_video_properties, get_dtype, get_stickman_line_connection

import queue


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
        

def ball_trajectory_and_bouncing(vid_path, raw_coordinates, vid_output_path, queuelength=15):

    # In order to draw the trajectory of tennis, we need to save the coordinate of previous 7 frames
    q = queue.deque()
    for i in range(0, queuelength):
        q.appendleft({'data':None, 'frame_id':None})

    frame_num     = 0

    video = cv2.VideoCapture(vid_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    print('fps : {}'.format(fps))
    output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc        = cv2.VideoWriter_fourcc(*'XVID')

    output_video = cv2.VideoWriter(vid_output_path, fourcc, fps, (output_width, output_height))
  

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


                PIL_image = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
                PIL_image = Image.fromarray(PIL_image)

                draw = ImageDraw.Draw(PIL_image)
                draw.line(ball_all, fill ="yellow", width = 4)

                img = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)

            output_video.write(img)
        
        else:
            break

  
        frame_num += 1

    output_video.release()
    video.release()



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
        
def interpolate_ball_trajectory(coords):
    # Remove Outliers 
    x, y = diff_xy(coords)
    remove_outliers(x, y, coords)

    # Interpolation
    coords = interpolation(coords)
    coords = np.array([[i[0], i[1]] for i in coords])

    return coords
    

def video_process(video_path, ball_detector):

    vid_name = video_path.split('/')[-1]
    

    # Load videos from videos path
    video = cv2.VideoCapture(video_path)

    # get videos properties
    fps, length, v_width, v_height = get_video_properties(video)
  
    # frame counter
    frame_i = 0

    total_time = 0

    # Loop over all frames in the videos
    while True:
        ret, frame = video.read()
        start_time = time.time()
        frame_i += 1

        if ret:
            ball_detector.detect_ball(frame)

        else:
            break
    

        total_time += (time.time() - start_time)
        print('Processing frame %d/%d  FPS %04f' % (frame_i, length, frame_i / total_time), '\r', end='')
        if not frame_i % 100:
            print('')

    coords = []
    for i in ball_detector.xy_coordinates:
        if i[0]==None:
            coords.append(None)

        else:
            
            coords.append(i)

    coords = interpolate_ball_trajectory(coords)

    
    vid_name.replace('.mkv', '.npy').replace('.mp4', '.npy')
    
    ball_trajectory_path = f"./output/{vid_name.replace('.mkv', '.npy').replace('.mp4', '.npy')}"
    output_vid_path      = f"./output/{vid_name}"
    

    ball_trajectory_and_bouncing(video_path, coords, output_vid_path)

    coords = np.array([[i[0] / v_width, i[1] / v_height] for i in coords], dtype=np.float)
    np.save(ball_trajectory_path, coords)


# def main():
#     os.makedirs('./outputs/', exist_ok=True)
#     videos_path = '/home/predator/Desktop/UPWORK/Tennis_tracking/tennis-tracking/segmented_video'
#     all_videos  = glob(videos_path+'/*')
    
#     ind = 0
#     # last_state = 42
#     ball_detector = BallDetector('./weights/model.3', out_channels=2)

#     for video_path in tqdm(all_videos):
#         ind += 1
#         print('index', ind)
#         # if ind > last_state:
#         ball_detector.xy_coordinates = np.array([[None, None], [None, None]])
#         video_process(video_path=video_path, ball_detector=ball_detector)
        

# if __name__ == "__main__":
#     main()
