import argparse
from sklearn.externals import joblib
import copy
from time import time
import numpy as np

from action_recognition.tracker import Tracker, TrackVisualiser
from action_recognition.detector import CaffeOpenpose
from action_recognition.analysis import PostProcessor, ChunkVisualiser


def main(args):
    classifier = joblib.load(args.classifier)
    detector = CaffeOpenpose(args.model_path)
    tracker = Tracker(detector)

    processor = PostProcessor()
    vis = TrackVisualiser()

    print("Classes: {}".format(classifier.classes_))

    track_people_start = time()
    for tracks, img, current_frame in tracker.video(args.video, True, True):
        print("Number of tracks: {}".format(len(tracks)))
        track_people_time = time() - track_people_start

        processor.tracks = [copy.deepcopy(t) for t in tracks]

        processor.post_process_tracks()
        chunks, _, _ = processor.chunk_tracks(30, 0, 30)

        print("Number of chunks: {}".format(len(chunks)))

        predict_people_start = time()
        if len(chunks) > 0:
            predictions = classifier.predict_proba(chunks)
            add_predictions_to_img(
                chunks, predictions, args.probability_threshold, classifier.classes_, vis, img)
        predict_people_time = time() - predict_people_start

        print("Predict time: {:.3f}, Track time: {:.3f}".format(
            predict_people_time, track_people_time))
        track_people_start = time()


def add_predictions_to_img(chunks, predictions, probability_threshold, classes, vis, img):
    for chunk, prediciton in zip(chunks, predictions):
        best_pred_i = np.argmax(prediciton)
        probability = prediciton[best_pred_i]
        pred_name = classes[best_pred_i]
        if probability > probability_threshold:
            position = tuple(chunk[-1, 0, :2].astype(np.int))
            print(pred_name)
            vis.draw_text(img, "{}: {:.3f}".format(pred_name, probability), position)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generates action predictions live given a video and a pre-trained classifier.')
    parser.add_argument('--classifier', type=str,
                        help='Path to a .pkl file with a pre-trained action recognition classifier.')
    parser.add_argument('--video', type=str,
                        help='Path to video file to predict actions for.')
    parser.add_argument('--model-path', type=str, default='../openpose/models/',
                        help='The model path for the caffe implementation.')
    parser.add_argument('--probability-threshold', type=float, default=0.8,
                        help='Threshold for how confident the model should be in each prediction.')

    args = parser.parse_args()
    main(args)
