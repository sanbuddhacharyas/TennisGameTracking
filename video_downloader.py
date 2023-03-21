import urllib.request
import yt_dlp
import subprocess

def download_video(url_link, save_location):
    urllib.request.urlretrieve(url_link, save_location) 

def download_video_from_youtube(link, save_loc):


    ydl_opts = {
        'outtmpl': f'{save_loc}'
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])

    # cmd = f"ffmpeg -i {save_loc}.webm {save_loc}.mp4"
    # subprocess.run(cmd,shell=True)

    # youtubeObject = YouTube(link)
    # youtubeObject.streams.filter(progressive=True)
    # youtubeObject = youtubeObject.streams.get_highest_resolution()
    # try:
    #     youtubeObject.download(save_loc)
    # except:
    #     print("An error has occurred")
    # print("Download is completed successfully")
