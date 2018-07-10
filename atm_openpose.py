from tracker import Tracker
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Openpose tracking system.')
    parser.add_argument('--video', type=str, default="media/video.avi",
                        help='The video to run tracking on.')

    args = parser.parse_args()

    model_path = '/path/to/models/'

    tracker = Tracker(tf_openpose=False, model_path=model_path)

    tracker.video(args.video, only_arms=True)
