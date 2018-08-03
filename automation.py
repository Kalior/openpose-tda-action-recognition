import os

from tracker import Tracker
from detector import CaffeOpenpose

model_path = "../openpose/models/"
detector = CaffeOpenpose(model_path)

for dirpath, dirnames, filenames in os.walk("../media/2018-08-02/"):

    for file in filenames:
        if file.endswith(".mp4"):
            out_dir = "../output/" + "/".join(dirpath.split("/")[2:])
            video_file = os.path.join(dirpath, file)
            print(out_dir)
            print(video_file)
            tracker = Tracker(detector=detector, out_dir=out_dir)
            tracker.video(video_file, False)
