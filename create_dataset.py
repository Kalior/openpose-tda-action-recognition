import logging
import argparse
import numpy as np
import os
import json
import cv2

from analysis import PostProcessor, Labelling
from tracker import TrackVisualiser


def main(args):
    if len(args.tracks_files) != len(args.videos):
        logging.warn("Number of track files does not correspond to number of videos.")
        return

    seconds_per_chunk = 20 / 30
    overlap_percentage = 0.5
    frames_per_chunk = 20
    number_of_keypoints = 18
    number_of_coordinates = 3
    all_chunks = np.empty((0, frames_per_chunk, number_of_keypoints, number_of_coordinates))
    all_frames = np.empty((0, frames_per_chunk))
    all_labels = np.empty((0,), dtype=str)
    all_videos = np.empty((0,), dtype=str)
    for i in range(len(args.tracks_files)):
        tracks_file = args.tracks_files[i]
        video = args.videos[i]
        logging.info("Processing video: {}".format(video))
        chunks, frames, labels = process_tracks(
            tracks_file, video, frames_per_chunk, overlap_percentage, seconds_per_chunk, args.filter_moving)
        videos = [video] * len(chunks)

        all_chunks = np.append(all_chunks, chunks, axis=0)
        all_frames = np.append(all_frames, frames, axis=0)
        all_labels = np.append(all_labels, np.array(list(labels.values())), axis=0)
        all_videos = np.append(all_videos, np.array(videos), axis=0)

    logging.info("Saving data to {}".format(args.out_file))
    if args.append and (os.path.isfile(args.out_file) or os.path.isfile(args.out_file + '.npz')):
        # Add extension if it isn't given
        if os.path.isfile(args.out_file):
            out_file = args.out_file
        elif os.path.isfile(args.out_file + '.npz'):
            out_file = args.out_file + '.npz'

        dataset_npz = np.load(out_file)

        all_chunks = load_and_append(dataset_npz, 'chunks', all_chunks)
        all_frames = load_and_append(dataset_npz, 'frames', all_frames)
        all_labels = load_and_append(dataset_npz, 'labels', all_labels)
        all_videos = load_and_append(dataset_npz, 'videos', all_videos)

    np.savez(args.out_file, chunks=all_chunks, frames=all_frames,
             labels=all_labels, videos=all_videos)


def load_and_append(npz, name, array):
    prev_array = npz[name]
    return np.append(prev_array, array, axis=0)


def process_tracks(tracks_file, video, target_frames_per_chunk, overlap_percentage, seconds_per_chunk, automatic_moving_filter):
    tracks_npz = np.load(tracks_file)
    np_tracks = tracks_npz['tracks']
    np_frames = tracks_npz['frames']

    logging.info("Combining, cleaning and removing tracks.")
    processor = PostProcessor()
    processor.create_tracks(np_tracks, np_frames)
    processor.post_process_tracks()

    capture = cv2.VideoCapture(video)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames_per_chunk_for_seconds = int(seconds_per_chunk * fps)
    logging.debug("Frames per chunks to get same #seconds: {}".format(
        frames_per_chunk_for_seconds))
    overlap = int(frames_per_chunk_for_seconds * overlap_percentage)

    logging.info("Chunking tracks.")
    chunks, frames = processor.chunk_tracks(
        frames_per_chunk_for_seconds, overlap, target_frames_per_chunk)
    logging.info("Identified {} chunks".format(chunks.shape[0]))

    if automatic_moving_filter:
        chunks, frames = filter_moving(chunks, frames)

    base_name = os.path.splitext(tracks_file)[0]
    labels_file = base_name + '.labels'
    if os.path.isfile(labels_file):
        with open(labels_file, 'r') as f:
            labels = json.load(f)
    else:
        labels = Labelling.label_chunks(chunks, frames, video, processor)
        j = json.dumps(labels)
        with open(labels_file, 'w') as f:
            f.write(j)

    # Filter the chunks that aren't/didn't get labelled
    logging.info("{} chunks labelled, and {} chunks removed".format(
        len(labels.keys()), chunks.shape[0] - len(labels.keys())))
    chunks = chunks[np.array(list(labels.keys()), dtype=np.int)]
    frames = frames[np.array(list(labels.keys()), dtype=np.int)]

    return chunks, frames, labels


def filter_moving(chunks, frames):
    chunks_before_filtering = chunks.shape[0]
    logging.info("Filtering out every path but the cachier standing still.")
    chunks, frames = processor.filter_moving_chunks(chunks, frames)
    logging.info("Automatically removed {} chunks".format(
        chunks_before_filtering - chunks.shape[0]))
    return chunks, frames

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset creation for analysis of tracks.')
    parser.add_argument('--videos', type=str, nargs='+',
                        help='The video from which the paths were generated.')
    parser.add_argument('--tracks-files', type=str, nargs='+',
                        help='The file with the saved tracks.')
    parser.add_argument('--out-file', type=str, default='dataset/dataset.npz',
                        help='The path to the file where the data will be saved')
    parser.add_argument('--append', action='store_true',
                        help='Specify if the data should be added to the out-file (if it exists) or overwritten.')
    parser.add_argument('--filter-moving', action='store_true',
                        help='Specify if you want to automatically filter chunks with large movement.')

    logging.basicConfig(level=logging.DEBUG)
    args = parser.parse_args()

    main(args)
