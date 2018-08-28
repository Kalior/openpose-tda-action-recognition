import argparse
import logging
import os

from action_recognition.tracker import Tracker
from action_recognition.detector import TFOpenpose, CaffeOpenpose


def main(args):
    if args.tf_openpose:
        detector = TFOpenpose()
    else:
        detector = CaffeOpenpose(model_path=args.model_path)

    videos = parse_path(args.video, args.out_directory, args.allowed_video_formats)

    for video, out_dir in videos:
        tracker = Tracker(detector=detector, out_dir=out_dir)
        tracker.video(video, args.draw_frames)


def parse_path(video, out_directory, allowed_video_formats):
    videos = []

    if os.path.isfile(video):
        videos.append((video, out_directory))
    elif os.path.isdir(video):
        for dirpath, dirnames, filenames in os.walk(video):
            for file in filenames:
                if any(file.endswith(format_) for format_ in allowed_video_formats):
                    out_dir = out_directory + "/".join(dirpath.split("/")[1:])
                    video_file = os.path.join(dirpath, file)
                    videos.append((video_file, out_dir))

    return videos

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('Generate tracks of people using OpenPose. '
                     'Each track is a [n_frames, n_keypoints, 3] numpy.ndarray which is predicted '
                     'as being a single person through several frames, saved to a .npz file.'))
    parser.add_argument('--video', type=str, default='media/video.avi',
                        help=('The video/folder to run tracking on. If folder, maintains the '
                              'hierarchy within that folder in the output.'))
    parser.add_argument('--tf-openpose', action='store_true',
                        help='Use to make the program use the tensorflow implementation of OpenPose.')
    parser.add_argument('--model-path', type=str, default='../openpose/models/',
                        help='The model path for the caffe implementation.')

    parser.add_argument('--out-directory', type=str, default='output',
                        help='Root directory to where the annotated video is saved.')
    parser.add_argument('--draw-frames', action='store_true',
                        help='Flag for if the frames with identified frames should be drawn or not.')
    parser.add_argument('--allowed-video-formats', type=str, nargs='+', default=['.mp4', '.avi'],
                        help='Used for filtering of videos if the parameter video is a directory.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(args)
