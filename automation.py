import os

from tracker import Tracker
from detector import CaffeOpenpose

model_path = "../openpose/models/"
for dirpath, dirnames, filenames in os.walk("media/"):

    for file in filenames:
        if file.endswith(".mp4"):
            out_dir = "output/" + "/".join(dirpath.split("/")[1:])
            video_file = os.path.join(dirpath, file)
            print(out_dir)
            print(video_file)
            detector = CaffeOpenpose(model_path)
            tracker = Tracker(detector=detector, only_track_arms=False, out_dir=out_dir)
            tracker.video(video_file)
