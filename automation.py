import os

from tracker import Tracker

model_path = "../openpose/models/"
for dirpath, dirnames, filenames in os.walk("media/checkout-better-angle"):

    for file in filenames:
        if file.endswith(".mp4"):
            out_dir = "output/" + "/".join(dirpath.split("/")[1:])
            video_file = os.path.join(dirpath, file)
            print(out_dir)
            print(video_file)
            tracker = Tracker(model_path=model_path, only_track_arms=True, out_dir=out_dir)
            tracker.video(video_file)
