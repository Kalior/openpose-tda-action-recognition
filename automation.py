import os

from action_recognition.tracker import Tracker
from action_recognition.detector import CaffeOpenpose

model_path = "../openpose/models/"
detector = CaffeOpenpose(model_path)

for dirpath, dirnames, filenames in os.walk("media/"):

    for file in filenames:
        if file.endswith(".mp4"):
            out_dir = "output/" + "/".join(dirpath.split("/")[1:])
            video_file = os.path.join(dirpath, file)
            print(out_dir)
            print(video_file)
            tracker = Tracker(detector=detector, out_dir=out_dir)
            tracker.video(video_file, False)
