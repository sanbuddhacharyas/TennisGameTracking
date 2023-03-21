import tensorflow as tf
import numpy as np
import cv2
import time
import os
from glob import glob
import subprocess
import shutil



from detection_new import analyize_tennis_game, find_game_in_video
from video_downloader import download_video, download_video_from_youtube

# def video_process(video_path):

    

def main():
    s = time.time()

    # try:
    #     shutil.rmtree('download')
    #     shutil.rmtree('game_output')

    # except:
    #     pass

    os.makedirs('videos', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    os.makedirs('CSV', exist_ok = True)
    os.makedirs('ball_npy', exist_ok = True)
    os.makedirs('ball_plots', exist_ok=True)
    os.makedirs('download', exist_ok=True)
    os.makedirs('game_output', exist_ok=True)

   

    # video_url = 'https://youtu.be/OXaEUUUJJ7s?t=1'
    # download_video_from_youtube(video_url, 'download/tennis_game')
    download_vid_path = glob("./download/*.webm")[0]
    
    # print(download_vid_path)
    # print("find_game_in_video")
    # find_game_in_video(vid_path=download_vid_path)

    # all_game = sorted(glob('./game_output/*.mp4'))
    
    for vid_path in ['./testing_data/Q0Vli051yX.mp4']:
        print(vid_path)
        
        analyize_tennis_game(vid_path)
            
        # except:
        #     pass

    
    print(f'Total computation time : {time.time() - s} seconds')


if __name__ == "__main__":
    main()
