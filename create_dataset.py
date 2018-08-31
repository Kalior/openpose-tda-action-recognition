import logging
import argparse
import numpy as np
import os
import json
import cv2
from sklearn.model_selection import train_test_split

from action_recognition.analysis import PostProcessor, Labelling
from action_recognition.tracker import TrackVisualiser


def main(args):
    tracks_files = parse_paths(args.tracks_files, 'tracks.npz')
    video_files = parse_paths(args.videos, '.mp4')

    if len(tracks_files) != len(video_files):
        logging.warn("Number of track files does not correspond to number of videos.")
        return

    seconds_per_chunk = 20 / 30
    overlap_percentage = 0.5

    all_chunks = []
    all_frames = []
    all_labels = []
    all_videos = []

    for tracks_file, video in zip(tracks_files, video_files):
        logging.info("Processing video: {} with tracks {}".format(video, tracks_file))

        chunks, frames, labels = process_tracks(
            tracks_file, video, args.frames_per_chunk, overlap_percentage,
            seconds_per_chunk)

        videos = np.array([video] * len(chunks))

        all_chunks.extend(chunks)
        all_frames.extend(frames)
        all_labels.extend(labels)
        all_videos.extend(videos)

    train, test = split_data(all_chunks, all_frames, all_labels, all_videos)

    extension_free = os.path.splitext(args.out_file)[0]
    train_name = extension_free + '-train.npz'
    test_name = extension_free + '-test.npz'

    append_and_save(*train, train_name)
    append_and_save(*test, test_name)


def parse_paths(paths, ok_ending):
    """
        Parses directories and regular files from the paths.
    """
    parsed_paths = []
    for path in paths:
        if os.path.isfile(path) and path.endswith(ok_ending):
            parsed_paths.append(path)
        elif os.path.isdir(path):
            for dirpath, dirnames, filenames in os.walk(path):
                for file in sorted(filenames):
                    if file.endswith(ok_ending):
                        file_path = os.path.join(dirpath, file)
                        parsed_paths.append(file_path)
    return parsed_paths


def append_and_save(chunks, frames, labels, videos, file_name):
    if args.append and os.path.isfile(file_name):
        dataset_npz = np.load(file_name)

        chunks = load_and_append(dataset_npz, 'chunks', chunks)
        frames = load_and_append(dataset_npz, 'frames', frames)
        labels = load_and_append(dataset_npz, 'labels', labels)
        videos = load_and_append(dataset_npz, 'videos', videos)

    logging.info("Saving data to {}".format(file_name))
    np.savez(file_name, chunks=chunks, frames=frames, labels=labels, videos=videos)


def load_and_append(npz, name, array):
    prev_array = npz[name]
    return np.append(prev_array, array, axis=0)


def split_data(chunks, frames, labels, videos):
    logging.info("Splitting data into test/train")
    train_chunks, test_chunks, train_labels, test_labels, \
        train_frames, test_frames, train_videos, test_videos = train_test_split(
            chunks, labels, frames, videos, test_size=0.2)

    return (train_chunks, train_frames, train_labels, train_videos), \
        (test_chunks, test_frames, test_labels, test_videos)


def process_tracks(tracks_file, video, target_frames_per_chunk, overlap_percentage, seconds_per_chunk):
    tracks_npz = np.load(tracks_file)
    np_tracks = tracks_npz['tracks']
    np_frames = tracks_npz['frames']

    logging.info("Combining, cleaning, and removing tracks.")
    processor = PostProcessor()
    processor.create_tracks(np_tracks, np_frames)
    processor.post_process_tracks()

    extension_free_tracks = os.path.splitext(tracks_file)[0]
    labels_file = extension_free_tracks + '-labels.json'

    if os.path.isfile(labels_file):
        with open(labels_file, 'r') as f:
            json_labels = json.load(f)
        chunks, frames, labels = Labelling().parse_labels(
            json_labels, target_frames_per_chunk, processor.tracks)
    else:
        extension_free_video = os.path.splitext(video)[0]
        timestamps_file = extension_free_video + '-timestamps.json'
        if os.path.isfile(timestamps_file):
            with open(timestamps_file, 'r') as f:
                timestamps = json.load(f)

            chunks, frames, labels, track_indicies = Labelling().pseudo_automatic_labelling(
                timestamps, target_frames_per_chunk, video, processor.tracks)
        else:
            chunks, frames, labels, track_indicies = Labelling().manual_labelling(
                video, processor, target_frames_per_chunk, overlap_percentage,
                seconds_per_chunk)

        Labelling().write_labels(chunks, frames, labels, track_indicies, labels_file)

    logging.info("{} chunks labelled".format(len(chunks)))

    return chunks, frames, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('Dataset creation/labelling for action recognition. '
                     'Allows for two different methods of labelling data, see '
                     'the documentation of the Labelling class for details.'))
    parser.add_argument('--videos', type=str, nargs='+',
                        help=('The videos/folders from which the paths were generated. '
                              'Needed for labelling of the data.'))
    parser.add_argument('--tracks-files', type=str, nargs='+',
                        help='The files/folders with the saved tracks.')
    parser.add_argument('--out-file', type=str, default='dataset/dataset',
                        help='The path to the file where the data will be saved.')
    parser.add_argument('--append', action='store_true',
                        help=('Specify if the data should be added to the out-file '
                              '(if it exists) or overwritten.'))

    parser.add_argument('--frames-per-chunk', type=int, default=20,
                        help=('The number of frames per chunk to divide tracks into. '
                              'If the data is already labelled with a different number of '
                              'frames per chunk, it will log a warning but try to extend '
                              'the labelled chunks to the given value.'))

    logging.basicConfig(level=logging.DEBUG)
    args = parser.parse_args()

    main(args)
