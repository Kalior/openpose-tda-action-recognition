from tracker import Tracker
import argparse
import logging
from detector import TFOpenpose, CaffeOpenpose


def main(args):
    if args.tf_openpose:
        detector = TFOpenpose()
    else:
        detector = CaffeOpenpose(model_path=args.model_path)

    tracker = Tracker(detector=detector,
                      only_track_arms=args.arm_tracking, out_dir=args.output_directory)
    tracker.video(args.video)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Openpose tracking system.')
    parser.add_argument('--video', type=str, default='media/video.avi',
                        help='The video to run tracking on.')
    parser.add_argument('--tf-openpose', action='store_true',
                        help='Use to make the program use the tensorflow implementation.')
    parser.add_argument('--model-path', type=str, default='../openpose/models/',
                        help='The model path for the caffe implementation.')
    parser.add_argument('--arm-tracking', action='store_true',
                        help='Use for arm/hand specific tracking.')
    parser.add_argument('--output-directory', type=str, default='output',
                        help='Directory to where the annotated video is saved.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    main(args)
