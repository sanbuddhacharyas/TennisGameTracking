import cv2
import imageio
import numpy as np
import tensorflow as tf

from detection import center_of_box
from utils_ import get_video_properties

def crop_players(frame, bbox, margin=20):
    return frame[bbox[1]-margin:bbox[3]+margin, bbox[0]-margin:bbox[2]+margin, :]


def get_stroke_predictions(video_path, stroke_model, strokes_frames, player_boxes):
    """
    Get the stroke prediction for all sections where we detected a stroke
    """
    predictions = {}
    video       = cv2.VideoCapture(video_path)
    fps, length, width, height = get_video_properties(video)

    print(fps, length)

    # For each stroke detected trim video part and predict stroke
    for frame_num in strokes_frames:
        game_play = make_video_segment(video, frame_num, player_boxes)

        print("game_play", len(game_play))
        if len(game_play)>20:
            game_play = frames_extraction(game_play, SEQUENCE_LENGTH=20)
            imageio.mimsave(f'./GIF/{frame_num}.gif', list((np.array(game_play)*255).astype(np.uint8)), fps=20)
            game_play = np.array(game_play)[np.newaxis, ...]
            

            pred   = stroke_model.predict(game_play)[0]
            cls_   = np.argmax(pred)

            if cls_ == 0:
                predictions[frame_num] = {'probs':pred, 'stroke': 'Backhand'}
            
            elif cls_ == 1:
                predictions[frame_num] = {'probs':pred, 'stroke': 'Service/Smash'}

            elif cls_ == 2:
                predictions[frame_num] = {'probs':pred, 'stroke': 'Forehand'}

        else:
            predictions[frame_num] = {'probs':[], 'stroke': 'None'}
    

    return predictions

def make_video_segment(video, frame_id, player_boxes):
    
    start    = max(0, (frame_id - 20))
    range_   = (frame_id + 15 - start)

    print("start", start)
    print("end", range_)

    h_max = []
    w_max = []

    max_frame = len(player_boxes)
    for i in range(range_):
        if max_frame > (start+i):
            bbox  = [player_boxes[start+i]] #tennis_tracking[tennis_tracking['Frame']==start+i][player_mode].to_numpy()
            if bbox[0][0] != None:
                h_max.append(int(bbox[0][3] - bbox[0][1]))
                w_max.append(int(bbox[0][2] - bbox[0][0]))

    h_margin = max(h_max)//2
    w_margin = max(w_max)//2

    video.set(cv2.CAP_PROP_POS_FRAMES, start)

    game_play = []
    for i in range(range_):
        res, frame = video.read()
        if res:
            
            if max_frame > (start+i):
                bbox  = [player_boxes[start+i]]#tennis_tracking[tennis_tracking['Frame']==start+i][player_mode].to_numpy()

                if len(bbox)!=0:
                    if bbox[0][0] != None:
                        player = crop_players(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), bbox[0])

                        box_center = center_of_box(bbox[0])
                        player     = frame[int(box_center[1] - h_margin): int(box_center[1] + h_margin),
                                int(box_center[0] - w_margin): int(box_center[0] + w_margin)].copy()

                        try:
                            game_play.append(player)

                        except:
                            player = game_play[-1]
                            game_play.append(player)


                    else:
                        if len(game_play)>0:
                            player = game_play[-1]
                            game_play.append(player)

    return game_play

def frames_extraction(frames_tennis, SEQUENCE_LENGTH = 20, IMAGE_HEIGHT= 64, IMAGE_WIDTH=64):
    '''
    This function will extract the required frames from a video after resizing and normalizing them.
    Args:
        video_path: The path of the video in the disk, whose frames are to be extracted.
    Returns:
        frames_list: A list containing the resized and normalized frames of the video.
    '''

    # Declare a list to store video frames.
    frames_list   = []

    # Get the total number of frames in the video.
    video_frames_count = len(frames_tennis)

    # Calculate the the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)

    num_frame_to_add   = SEQUENCE_LENGTH - video_frames_count

    if num_frame_to_add > 0:
        add_frame_sequence = int(np.floor(SEQUENCE_LENGTH / num_frame_to_add))
       

    vid_counter = 0
    # Iterate through the Video Frames.
    for frame_counter in range(SEQUENCE_LENGTH):

        if(num_frame_to_add<=0):
            
            frame = frames_tennis[frame_counter * skip_frames_window]

            # Resize the Frame to fixed height and width.
            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            
            # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
            normalized_frame = resized_frame / 255
            
            # Append the normalized frame into the frames list
            frames_list.append(normalized_frame)


        else:
            if ((frame_counter+1)%add_frame_sequence)!=0:
                    # # Set the current frame position of the video.
                # video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

                # Reading the frame from the video. 
                frame = frames_tennis[vid_counter]

                # Resize the Frame to fixed height and width.
                resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
                
                # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
                normalized_frame = resized_frame / 255
                
                # Append the normalized frame into the frames list
                frames_list.append(normalized_frame)

                vid_counter += 1

            elif len(frames_list)>0:
                frames_list.append(frames_list[-1])
    
    # Return the frames list.
    return frames_list

if __name__ == '__main__':
    stroke_model = tf.keras.models.load_model('./checkpoints/stroke_classification.h5')
    print(stroke_model.summary())
    print("Model_loaded")
    video_path  = './action_recognition_player/9J0GI0R000_1490_P2.gif'
    video       = cv2.VideoCapture(video_path)

    stroke_frames = []
    while True:
        ret, img = video.read()
        if ret:
            stroke_frames.append(img)

        else:
            break
    
    print(len(stroke_frames))
    stroke_frames = np.array(frames_extraction(stroke_frames, SEQUENCE_LENGTH=20))[np.newaxis, ...]

    print(stroke_frames.shape)

    prediction    = stroke_model.predict(stroke_frames)[0]
    pred          = np.argmax(prediction)

    print(prediction, pred)
