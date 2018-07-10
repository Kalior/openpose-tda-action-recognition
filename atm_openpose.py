from tracker import Tracker
import argparse

if __name__ == '__main__':
    model_path = '/home/kalior/projects/fujitsu-CoE/openpose/models/'

    tracker = Tracker(tf_openpose=True, model_path=model_path)

    tracker.video("media/checkout-2.mp4", only_arms=True)
