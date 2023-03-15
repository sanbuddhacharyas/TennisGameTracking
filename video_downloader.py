import urllib.request
from pytube import YouTube

def download_video(url_link, save_location):
    urllib.request.urlretrieve(url_link, save_location) 

def download_video_from_youtube(link, save_loc):

    youtubeObject = YouTube(link)
    youtubeObject.streams.filter(progressive=True)
    youtubeObject = youtubeObject.streams.get_highest_resolution()
    try:
        youtubeObject.download(save_loc)
    except:
        print("An error has occurred")
    print("Download is completed successfully")
