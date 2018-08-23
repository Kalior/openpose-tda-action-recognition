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
    #  Copy video file so we can create multiple different videos
    # with it as base simultaneously.
    tmp_video_file = "output/tmp" + video_ending
    copyfile(args.video, tmp_video_file)

    classifier = joblib.load(args.classifier)
    detector = CaffeOpenpose(args.model_path)
    tracker = Tracker(detector, out_dir=args.out_directory)

    processor = PostProcessor()

    classes = classifier.classes_
    logging.info("Classes: {}".format(classes))

    track_people_start = time()
    valid_predictions = []
    for tracks, img, current_frame in tracker.video(args.video, True, True):
        #  Only predict every 5:th frame.  Need to figure out a way to
        # be more systematical about when to predict and which chunks.
        if current_frame % 5 != 0 or len(tracks) <= 0:
            write_predictions(valid_predictions, img)
            continue

        logging.debug("Number of tracks: {}".format(len(tracks)))
        track_people_time = time() - track_people_start

        #  Extract the latest frames, as we don't want to copy
        # too much data here, and we've already predicted for the rest
        processor.tracks = [copy.deepcopy(t.copy(-50)) for t in tracks]
        processor.post_process_tracks()
        tracks = processor.tracks

        predict_people_start = time()

        predictions = [predict_per_track(t, classifier) for t in tracks]

        logging.info("Predictions: " + ", ".join(
            ["{}: {:.3f}".format(*get_best_pred(prediction, classes))
             for _, _, prediction in predictions]))

        valid_predictions = filter_bad_predictions(
            predictions, args.probability_threshold, classes)

        predict_people_time = time() - predict_people_start

        write_predictions(valid_predictions, img)
        save_predictions(valid_predictions, args.video, tmp_video_file, args.out_directory)

        logging.debug("Predict time: {:.3f}, Track time: {:.3f}".format(
            predict_people_time, track_people_time))
        track_people_start = time()


def predict_per_track(track, classifier):
    all_chunks = []
    all_frames = []
    divisions = [(50, 0), (30, 10), (25, 0), (20, 5)]
    for frames_per_chunk, overlap in divisions:
        chunks, chunk_frames = track.divide_into_chunks(frames_per_chunk, overlap)
        if len(chunks) > 0:
            for chunk in chunks:
                all_chunks.append(chunk)
            for frames in chunk_frames:
                all_frames.append(frames)

    if len(all_chunks) > 0:
        predictions = classifier.predict_proba(all_chunks)
        average_prediction = np.mean(predictions, axis=0)
        print(predictions)
        print(average_prediction)
        return all_chunks[0], all_frames[0], average_prediction
    else:
        return None, None, [0] * len(classifier.classes_)


def write_predictions(valid_predictions, img):
    for label, probability, position, chunk, frames in valid_predictions:
        TrackVisualiser().draw_text(img, "{}: {:.3f}".format(label, probability), position)


def save_predictions(valid_predictions, video_name, video, out_directory):
    for i, (label, probability, position, chunk, frames) in enumerate(valid_predictions):
        write_chunk_to_file(video_name, video, frames, chunk, label, out_directory, i)


def get_best_pred(prediction, classes):
    best_pred_i = np.argmax(prediction)
    probability = prediction[best_pred_i]
    label = classes[best_pred_i]
    return label, probability


def filter_bad_predictions(zipped, threshold, classes):
    valid_predictions = []
    for chunk, frames, prediction in zipped:
        label, probability = get_best_pred(prediction, classes)
        if probability > threshold:
            position = tuple(chunk[-1, 0, :2].astype(np.int))
            prediction_tuple = (label, probability, position, chunk, frames)
            valid_predictions.append(prediction_tuple)

    return valid_predictions


def write_chunk_to_file(video_name, video, frames, chunk, label, out_dir, i):
    _, video_name = os.path.split(video_name)
    video_name, _ = os.path.splitext(video_name)
    file_name = "{}-{}-{}.avi".format(video_name, frames[-1], i)
    out_file = os.path.join(out_dir, file_name)
    ChunkVisualiser().chunk_to_video_scene(video, chunk, out_file, frames, label)


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
