import tensorflow as tf
import numpy as np
import cv2
import time
import os
from glob import glob



from detection_new import find_game_in_video, analyize_tennis_game
from video_downloader import download_video, download_video_from_youtube

# def video_process(video_path):

    
def main():
    s = time.time()
    os.makedirs('videos', exist_ok=True)

    try:
        os.removedirs('download')
    except:
        pass

    video_url = 'https://youtu.be/_-Fiw5bdAw4'
    # download_video_from_youtube(video_url, 'download')
    # download_vid_path = glob("./download/*")[0]
    
    # find_game_in_video(vid_path=download_vid_path)
    
    for vid_path in ['./testing_data/T147Ji7BsQ.mp4']:
        print(vid_path)
        analyize_tennis_game(vid_path)

    
    print(f'Total computation time : {time.time() - s} seconds')


if __name__ == "__main__":
    main()
