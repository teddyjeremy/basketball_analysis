from utils import read_video, save_video
from trackers import PlayerTracker
def main():

    #read video frames from input video
    video_frames = read_video("input_videos/video_1.mp4")

    #Initialize player tracker 
    player_tracker = PlayerTracker("models/player_detector.pt")

    #Run tracker
    player_tracks = player_tracker.get_object_tracks(video_frames)

    #save video frames to output video
    save_video(video_frames, "output_videos/output_video.avi")


    
if __name__ == "__main__":
    main()