from tracker import Tracker
import argparse


def main(args):
    if args.tf_openpose:
        tracker = Tracker(tf_openpose=args.tf_openpose)
    else:
        tracker = Tracker(model_path=args.model_path)

    tracker.video(args.video, only_arms=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Openpose tracking system.')
    parser.add_argument('--video', type=str, default='media/video.avi',
                        help='The video to run tracking on.')
    parser.add_argument('--tf-openpose', action='store_true',
                        help='Use to make the program use the tensorflow implementation.')
    parser.add_argument('--model-path', type=str, default='../openpose/models/',
                        help='The model path for the caffe implementation.')

    args = parser.parse_args()

    main(args)
