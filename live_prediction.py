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

    processor = PostProcessor()

    classes = classifier.classes_
    logging.info("Classes: {}".format(classes))

    track_people_start = time()
    valid_predictions = []
    for tracks, img, current_frame in tracker.video_generator(args.video, args.draw_frames):
        #  Don't predict every frame, not enough has changed for it to be valuable.
        if current_frame % 20 != 0 or len(tracks) <= 0:
            write_predictions(valid_predictions, img)
            continue

        # We only care about recently updated tracks.
        tracks = [track for track in tracks
                  if track.recently_updated(current_frame)]

        logging.debug("Number of tracks: {}".format(len(tracks)))
        track_people_time = time() - track_people_start

        #  Extract the latest frames, as we don't want to copy
        # too much data here, and we've already predicted for the rest
        processor.tracks = [copy.deepcopy(t.copy(-50)) for t in tracks]
        processor.post_process_tracks()

        predict_people_start = time()

        predictions = [predict_per_track(t, classifier) for t in processor.tracks]

        logging.info("Predictions: " + ", ".join(
            ["{}: {:.3f}".format(*get_best_pred(prediction, classes))
             for _, _, prediction in predictions]))

        valid_predictions = filter_bad_predictions(
            predictions, args.confidence_threshold, classes)
        save_predictions_to_track(predictions, classes, tracks, current_frame)

        predict_people_time = time() - predict_people_start

        not_stopped = [predict_no_stop(track, args.confidence_threshold) for track in tracks]
        [valid_predictions.append(t) for not_stopped, t in not_stopped if not_stopped]

        if not_stopped:
            logging.info("Not stopped: " + ", ".join(
                [str(i) for i, (prediction, _) in enumerate(not_stopped) if prediction]))

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


def get_best_pred(prediction, classes):
    best_pred_i = np.argmax(prediction)
    confidence = prediction[best_pred_i]
    label = classes[best_pred_i]
    return label, confidence


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


def write_chunk_to_file(video_name, video, frames, chunk, label, out_dir, i):
    _, video_name = os.path.split(video_name)
    video_name, _ = os.path.splitext(video_name)
    file_name = "{}-{}-{}-{}.avi".format(video_name, frames[-1], i, label)
    out_file = os.path.join(out_dir, file_name)
    ChunkVisualiser().chunk_to_video_scene(video, chunk, out_file, frames, label)


def predict_no_stop(track, confidence_threshold, stop_threshold=10):
    if len(track) < 50:
        return False, ()

    speed_prediction, prediction_tuple = speed_no_stop_prediction(track, stop_threshold)
    classifier_prediction = classifier_predict_no_stop(track, confidence_threshold)

    if speed_prediction and classifier_prediction:
        prediction_tuple = ("Both " + prediction_tuple[0], *prediction_tuple[1:])
    elif speed_prediction:
        prediction_tuple = ("Speed " + prediction_tuple[0], *prediction_tuple[1:])
    elif classifier_prediction:
        prediction_tuple = ("Classifier " + prediction_tuple[0], *prediction_tuple[1:])

    return speed_prediction or classifier_prediction, prediction_tuple


def classifier_predict_no_stop(track, confidence_threshold):
    if len(track.predictions) == 0:
        return False

    constant_moving = all(prediction['label'] == 'moving' and
                          prediction['confidence'] > confidence_threshold or
                          prediction['confidence'] < confidence_threshold
                          for prediction in list(track.predictions.values())[-20:])
    return constant_moving


def speed_no_stop_prediction(track, stop_threshold):
    #  Only check last 200 frames as person could have been doing something else
    # before that.  Makes the prediction a bit fragile.
    track = track.copy(-200)
    chunks, chunk_frames = track.divide_into_chunks(len(track) - 1, 0)
    keypoint_speed = transforms.Speed().fit_transform(chunks)[0]
    frame_speed = np.mean(keypoint_speed[:, :, :2], axis=1)
    frame_speed = np.linalg.norm(frame_speed, axis=1)

    # Find first index where there is movement. Count from there.
    first_movement_index = np.where(frame_speed > stop_threshold)[0][0]
    n_movement_frames = np.count_nonzero(frame_speed[first_movement_index:] > stop_threshold)

    #  Calculate how many of the last moving frames have to have had movement
    # for us to predict the person did not stop to do anything.
    # The 50 here is arbitrary and might make the prediction a bit fragile
    n_movement_frames_for_no_stop = len(track) - first_movement_index - 50

    position = tuple(chunks[0, -1, 1, :2].astype(np.int))
    prediction_tuple = ("not stopped", 1, position, chunks[0], chunk_frames[0])

    return n_movement_frames >= n_movement_frames_for_no_stop, prediction_tuple

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generates action predictions live given a video and a pre-trained classifier.')
    parser.add_argument('--classifier', type=str,
                        help='Path to a .pkl file with a pre-trained action recognition classifier.')
    parser.add_argument('--video', type=str,
                        help='Path to video file to predict actions for.')
    parser.add_argument('--model-path', type=str, default='../openpose/models/',
                        help='The model path for the caffe implementation.')
    parser.add_argument('--confidence-threshold', type=float, default=0.8,
                        help='Threshold for how confident the model should be in each prediction.')

    parser.add_argument('--draw-frames', action='store_true',
                        help='Flag for if the frames with identified frames should be drawn or not.')

    parser.add_argument('--out-directory', type=str, default='output/prediction',
                        help=('Output directory to where the processed video and identified '
                              'chunks are saved.'))

    logging.basicConfig(level=logging.DEBUG)

    args = parser.parse_args()
    main(args)
