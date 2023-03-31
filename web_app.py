import streamlit as st
import time
import shutil
import os

from random import randint
from glob import glob

from detection_new import analyize_tennis_game, find_game_in_video
from video_downloader import download_video, download_video_from_youtube

def get_state():
    st.session_state.widget_key = str(randint(1000, 100000000))

STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""

title_alignment="""
<style>
GAME SET MATH {
  text-align: center
}
</style>
"""

try:
    shutil.rmtree('download')
    shutil.rmtree('game_output')

except:
    pass

os.makedirs('videos', exist_ok=True)
os.makedirs('output', exist_ok=True)
os.makedirs('CSV', exist_ok = True)
os.makedirs('ball_npy', exist_ok = True)
os.makedirs('ball_plots', exist_ok=True)
os.makedirs('download', exist_ok=True)
os.makedirs('game_output', exist_ok=True)
os.makedirs('GIF', exist_ok=True)

st.markdown("<h1 style='text-align: center; color: black;'>GAME SET MATH &#129358</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: black;'>Tennis Game Analysis </h2>", unsafe_allow_html=True)


show_file = st.empty()
show_file.image('./image/tennis_wallpaper.jpg')


if 'widget_key' not in st.session_state:
    st.session_state.widget_key = str(randint(1000, 100000000))
          
col1, col2 = st.columns([1,1])

with col1:
    num_game_play = st.text_input("Number of game play")

with col2:
    url = st.text_input("Insert YouTube URL", key= st.session_state.widget_key)

if url != '':
    completed = 0
    progress_text = "Downloading Video Please wait."
    my_bar = st.progress(completed, text=progress_text)
    download_video_from_youtube(url, 'download/tennis_game')

    for percent_complete in range(10):
        time.sleep(0.5)
        completed += 1
        my_bar.progress(completed, text=progress_text)

    my_bar.progress(completed, text='Segmenting Game From the Video...')

    download_vid_path = glob("./download/*.webm")[0]

    print('num_game_play', num_game_play)
    if num_game_play == '':
        num_game_play = 1
    
    else:
        num_game_play = int(num_game_play)

    num_game_play = 1
    find_game_in_video(download_vid_path, num_game_play)
    for percent_complete in range(10):
        time.sleep(0.7)
        completed += 1
        my_bar.progress(completed, text='Segmenting Game From the Video...')

    my_bar.progress(completed, text='Analyzing Video, Please wait...')
    all_game = sorted(glob('./game_output/*.mp4'))

    total_games       = len(all_game)
    one_game_segment  = (100 - completed)//total_games

    for ind, vid_path in enumerate(all_game):

        try:
            analyize_tennis_game(vid_path, my_bar, ind, one_game_segment, completed)
            
            vid_name    = vid_path.split('/')[-1].split('.')[0]
            video_file  = open(f'./output/{vid_name}.webm', 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)

            col1, col2 = st.columns([1,1])
            with col1:
                with open(f'./output/{vid_name}.webm', "rb") as file:
                            btn = st.download_button(
                                    label="Download Video",
                                    data=file,
                                    file_name=f'{vid_name}.webm',
                                    on_click = get_state
                                )

            with col2:
                with open(f'./CSV/{vid_name}.xlsx', "rb") as file:
                            btn = st.download_button(
                                    label="Download Excel File",
                                    data=file,
                                    file_name=f'{vid_name}.xlsx',
                                    on_click = get_state
                                )
        except:
            continue
       

    url = ''  
        # except:
        #     pass

    # my_bar = st.progress(10, text=)
   
   

# progress_text = "Operation in progress. Please wait."
# my_bar = st.progress(0, text=progress_text)

# for percent_complete in range(100):
#     time.sleep(0.1)
#     my_bar.progress(percent_complete + 1, text=progress_text)