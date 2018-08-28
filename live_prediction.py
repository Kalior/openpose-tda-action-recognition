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
from action_recognition import transforms


def main(args):
    os.makedirs(args.out_directory, exist_ok=True)

    _, video_ending = os.path.splitext(args.video)
    #  Copy video file so we can create multiple different videos
    # with it as base simultaneously.
    tmp_video_file = "output/tmp" + video_ending
    copyfile(args.video, tmp_video_file)

    classifier = joblib.load(args.classifier)

    detector = CaffeOpenpose(args.model_path)
    tracker = Tracker(detector, out_dir=args.out_directory)

    logging.info("Classes: {}".format(classifier.classes_))

    valid_predictions = []
    track_people_start = time()
    for tracks, img, current_frame in tracker.video_generator(args.video, args.draw_frames):
        #  Don't predict every frame, not enough has changed for it to be valuable.
        if current_frame % 20 != 0 or len(tracks) <= 0:
            write_predictions(valid_predictions, img)
            continue

        # We only care about recently updated tracks.
        tracks = [track for track in tracks
                  if track.recently_updated(current_frame)]

        track_people_time = time() - track_people_start
        logging.debug("Number of tracks: {}".format(len(tracks)))

        predict_people_start = time()

        valid_predictions = predict(tracks, classifier, current_frame, args.confidence_threshold)

        predict_people_time = time() - predict_people_start

        write_predictions(valid_predictions, img)
        save_predictions(valid_predictions, args.video, tmp_video_file, args.out_directory)

        logging.info("Predict time: {:.3f}, Track time: {:.3f}".format(
            predict_people_time, track_people_time))
        track_people_start = time()


def predict(tracks, classifier, current_frame, confidence_threshold):
    #  Extract the latest frames, as we don't want to copy
    # too much data here, and we've already predicted for the rest
    processor = PostProcessor()
    processor.tracks = [copy.deepcopy(t.copy(-50)) for t in tracks]
    processor.post_process_tracks()

    predictions = [predict_per_track(t, classifier) for t in processor.tracks]

    valid_predictions = filter_bad_predictions(
        predictions, confidence_threshold, classifier.classes_)
    save_predictions_to_track(predictions, classifier.classes_, tracks, current_frame)

    no_stop_predictions = [predict_no_stop(track, confidence_threshold)
                           for track in tracks]

    for t in [t for p, t in no_stop_predictions if p]:
        valid_predictions.append(t)

    log_predictions(predictions, no_stop_predictions, classifier.classes_)

    return valid_predictions


def predict_per_track(track, classifier):
    all_chunks = []
    all_frames = []
    divisions = [(50, 0), (30, 10), (25, 0), (20, 5)]
    for frames_per_chunk, overlap in divisions:
        chunks, chunk_frames = track.divide_into_chunks(frames_per_chunk, overlap)
        if len(chunks) > 0:
            all_chunks.append(chunks[-1])
            all_frames.append(chunk_frames[-1])

    if len(all_chunks) > 0:
        predictions = classifier.predict_proba(all_chunks)
        average_prediction = np.amax(predictions, axis=0)
        return all_chunks[0], all_frames[0], average_prediction
    else:
        return None, None, [0] * len(classifier.classes_)


def write_predictions(valid_predictions, img):
    for label, confidence, position, _, _ in valid_predictions:
        TrackVisualiser().draw_text(img, "{}: {:.3f}".format(label, confidence), position)


def save_predictions(valid_predictions, video_name, video, out_directory):
    for i, (label, _, _, chunk, frames) in enumerate(valid_predictions):
        write_chunk_to_file(video_name, video, frames, chunk, label, out_directory, i)


def filter_bad_predictions(predictions, threshold, classes):
    valid_predictions = []
    for chunk, frames, prediction in predictions:
        label, confidence = get_best_pred(prediction, classes)
        if confidence > threshold:
            position = tuple(chunk[-1, 0, :2].astype(np.int))
            prediction_tuple = (label, confidence, position, chunk, frames)
            valid_predictions.append(prediction_tuple)

    return valid_predictions


def save_predictions_to_track(predictions, classes, tracks, current_frame):
    for t, (_, _, prediction) in zip(tracks, predictions):
        label, confidence = get_best_pred(prediction, classes)
        t.add_prediction(label, confidence, current_frame)


def get_best_pred(prediction, classes):
    best_pred_i = np.argmax(prediction)
    confidence = prediction[best_pred_i]
    label = classes[best_pred_i]
    return label, confidence


def write_chunk_to_file(video_name, video, frames, chunk, label, out_dir, i):
    _, video_name = os.path.split(video_name)
    video_name, _ = os.path.splitext(video_name)
    file_name = "{}-{}-{}-{}.avi".format(video_name, frames[-1], i, label)
    out_file = os.path.join(out_dir, file_name)
    ChunkVisualiser().chunk_to_video_scene(video, chunk, out_file, frames, label)


def predict_no_stop(track, confidence_threshold, stop_threshold=10):
    classifier_prediction = classifier_predict_no_stop(track, confidence_threshold)

    #  Copy last 200 frames to chunk for visusalisation.
    track = track.copy(-200)
    chunks, chunk_frames = track.divide_into_chunks(len(track) - 1, 0)

    position = tuple(chunks[0, -1, 1, :2].astype(np.int))
    prediction_tuple = ("Has not stopped", confidence, position, chunks[0], chunk_frames[0])
    return confidence > confidence_threshold, prediction_tuple


def classifier_predict_no_stop(track, confidence_threshold):
    # If there haven't been that many predictions, we can't say anything.
    if len(track.predictions) < 5:
        return 0

    number_moving = sum(prediction['label'] == 'moving' and
                        prediction['confidence'] > confidence_threshold
                        for prediction in list(track.predictions.values())[-20:])

    return number_moving / len(track.predictions)


def log_predictions(predictions, no_stop_predictions, classes):
    prints = []
    for _, _, prediction in predictions:
        prints.append(get_best_pred(prediction, classes))

    if no_stop_predictions:
        for label, confidence, _, _, _ in [t for p, t in no_stop_predictions if p]:
            prints.append((label, confidence))

    logging.info("Predictions: " + ", ".join(
        ["{}: {:.3f}".format(*t)
         for t in prints]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('Generates action predictions live given a video and a pre-trained classifier. '
                     'Uses Tracker.tracker.video_generator which yields every track every frame, '
                     'from which it predicts the class of action using the pre-trained classifier. '
                     'To get a better prediction, it takes the latest 50, 30, 25, and 20 frames '
                     'as chunks and selects the likliest prediction among the five * n_classes. '
                     'It also predicts if a person has not stopped moving (e.g. if they are moving '
                     'through a self-checkout area without scanning anything) by checking if '
                     'a proportion of the latest identified actions for a track/person is moving.'))

    parser.add_argument('--classifier', type=str,
                        help='Path to a .pkl file with a pre-trained action recognition classifier.')
    parser.add_argument('--video', type=str,
                        help='Path to video file to predict actions for.')
    parser.add_argument('--model-path', type=str, default='../openpose/models/',
                        help='The model path for OpenPose.')
    parser.add_argument('--confidence-threshold', type=float, default=0.8,
                        help='Threshold for how confident the model should be in each prediction.')

    parser.add_argument('--draw-frames', action='store_true',
                        help='Flag for if the frames with identified frames should be drawn or not.')

    parser.add_argument('--out-directory', type=str, default='output/prediction',
                        help=('Output directory to where the processed video and identified '
                              'chunks are saved.'))

    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    main(args)
