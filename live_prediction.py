import argparse
from sklearn.externals import joblib
import copy
from time import time

from action_recognition.tracker import Tracker
from action_recognition.detector import CaffeOpenpose
from action_recognition.analysis import PostProcessor


def main(args):
    classifier = joblib.load(args.classifier)
    detector = CaffeOpenpose(args.model_path)
    tracker = Tracker(detector)

    processor = PostProcessor()

    track_people_start = time()
    for tracks, current_frame in tracker.video(args.video, True, True):
        print("Number of tracks: {}".format(len(tracks)))
        track_people_time = time() - track_people_start

        processor.tracks = [copy.copy(t) for t in tracks]

        processor.post_process_tracks()
        chunks, _, _ = processor.chunk_tracks(30, 0, 30)

        print("Number of chunks: {}".format(len(chunks)))

        predict_people_start = time()
        if len(chunks) > 0:
            print(classifier.predict(chunks))
        predict_people_time = time() - predict_people_start

        print("Predict time: {:.3f}, Track time: {:.3f}".format(
            predict_people_time, track_people_time))
        track_people_start = time()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generates action predictions live given a video and a pre-trained classifier.')
    parser.add_argument('--classifier', type=str,
                        help='Path to a .pkl file with a pre-trained action recognition classifier.')
    parser.add_argument('--video', type=str,
                        help='Path to video file to predict actions for.')
    parser.add_argument('--model-path', type=str, default='../openpose/models/',
                        help='The model path for the caffe implementation.')

    args = parser.parse_args()
    main(args)
