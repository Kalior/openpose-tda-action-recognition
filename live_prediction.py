import argparse
from sklearn.externals import joblib
import copy
from time import time
import numpy as np
import logging
import os
from shutil import copyfile

from action_recognition.tracker import Tracker, TrackVisualiser
from action_recognition.detector import CaffeOpenpose
from action_recognition.analysis import PostProcessor, ChunkVisualiser


def main(args):
    _, video_ending = os.path.splitext(args.video)
    # Copy video file so we can create multiple different videos
    # with it as base simultaneously.
    tmp_video_file = "output/tmp" + video_ending
    copyfile(args.video, tmp_video_file)

    classifier = joblib.load(args.classifier)
    detector = CaffeOpenpose(args.model_path)
    tracker = Tracker(detector, out_dir=args.out_directory)

    processor = PostProcessor()
    vis = TrackVisualiser()

    classes = classifier.classes_
    logging.info("Classes: {}".format(classes))

    track_people_start = time()
    for tracks, img, current_frame in tracker.video(args.video, True, True):
        logging.debug("Number of tracks: {}".format(len(tracks)))
        track_people_time = time() - track_people_start

        processor.tracks = [copy.deepcopy(t) for t in tracks]

        processor.post_process_tracks()
        chunks, chunk_frames, _ = processor.chunk_tracks(30, 0, 30)

        logging.debug("Number of chunks: {}".format(len(chunks)))

        predict_people_start = time()
        if len(chunks) > 0:
            predictions = classifier.predict_proba(chunks)

            logging.info("Predictions: " + ", ".join(
                ["{}: {:.3f}".format(*get_best_pred(prediction, classes))
                 for prediction in predictions]))

            zipped = zip(chunks, chunk_frames, predictions)

            add_predictions_to_img(
                zipped, args.probability_threshold,
                classes, img, args.video, tmp_video_file, args.out_directory)
        predict_people_time = time() - predict_people_start

        logging.debug("Predict time: {:.3f}, Track time: {:.3f}".format(
            predict_people_time, track_people_time))
        track_people_start = time()


def get_best_pred(prediction, classes):
    best_pred_i = np.argmax(prediction)
    probability = prediction[best_pred_i]
    label = classes[best_pred_i]
    return label, probability


def add_predictions_to_img(zipped, threshold, classes, img, video, video_copy, out_dir):
    for i, (chunk, frames, prediction) in enumerate(zipped):
        label, probability = get_best_pred(prediction, classes)
        if probability > threshold:
            position = tuple(chunk[-1, 0, :2].astype(np.int))
            TrackVisualiser().draw_text(img, "{}: {:.3f}".format(label, probability), position)

            _, video_name = os.path.split(video)
            video_name, _ = os.path.splitext(video_name)
            file_name = "{}-{}-{}.avi".format(video_name, frames[-1], i)
            out_file = os.path.join(out_dir, file_name)
            ChunkVisualiser().chunk_to_video_scene(video_copy, chunk, out_file, frames, label)


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

    parser.add_argument('--out-directory', type=str, default='output/prediction',
                        help=('Output directory to where the processed video and identified '
                              'chunks are saved.'))

    logging.basicConfig(level=logging.DEBUG)

    args = parser.parse_args()
    main(args)
