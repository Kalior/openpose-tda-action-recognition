import logging
import argparse
import numpy as np
import os
import json

from analysis import PostProcessor, Labelling
from tracker import TrackVisualiser


def main(args):
    if len(args.tracks_files) != len(args.videos):
        logging.warn("Number of track files does not correspond to number of videos.")
        return

    frames_per_chunk = 20
    overlap = 10
    number_of_keypoints = 18
    number_of_coordinates = 3
    all_chunks = np.empty((0, frames_per_chunk, number_of_keypoints, number_of_coordinates))
    all_frames = np.empty((0,))
    all_labels = np.empty((0,), dtype=str)
    all_videos = np.empty((0,), dtype=str)
    for i in range(len(args.tracks_files)):
        tracks_file = args.tracks_files[i]
        video = args.videos[i]
        logging.info("Processing video: {}".format(video))
        chunks, frames, labels = process_tracks(
            tracks_file, video, frames_per_chunk, overlap, args.filter_moving)
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
        prev_chunks = dataset_npz['chunks']
        all_chunks = np.append(prev_chunks, all_chunks, axis=0)
        prev_frames = dataset_npz['frames']
        all_frames = np.append(prev_frames, all_frames, axis=0)
        prev_labels = dataset_npz['labels']
        all_labels = np.append(prev_labels, all_labels, axis=0)
        prev_videos = dataset_npz['videos']
        all_videos = np.append(prev_videos, all_videos, axis=0)

    np.savez(args.out_file, chunks=all_chunks, frames=all_frames,
             labels=all_labels, videos=all_videos)


def process_tracks(tracks_file, video, frames_per_chunk, overlap, automatic_moving_filter):
    tracks_npz = np.load(tracks_file)
    np_tracks = tracks_npz['tracks']
    np_frames = tracks_npz['frames']

    logging.info("Combining, cleaning and removing tracks.")
    processor = PostProcessor()
    processor.create_tracks(np_tracks, np_frames)
    processor.post_process_tracks()

    logging.info("Chunking tracks.")
    chunks, frames = processor.chunk_tracks(frames_per_chunk, overlap)
    logging.info("Identified {} chunks".format(len(chunks)))
    if automatic_moving_filter:
        chunks_before_filtering = chunks.shape[0]
        logging.info("Filtering out every path but the cachier standing still.")
        chunks, frames = processor.filter_moving_chunks(chunks, frames)
        logging.info("Automatically removed {} chunks".format(
            chunks_before_filtering - chunks.shape[0]))

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

    # last_frame = np.amax([f[-1] for f in np_frames])
    # visualiser = TrackVisualiser()
    # filtered_tracks = processor.chunks_to_tracks(chunks, frames)
    # visualiser.draw_video_with_tracks(filtered_tracks, video, last_frame)

    return chunks, frames, labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset creation for analysis of tracks.')
    parser.add_argument('--videos', type=str, nargs='+',
                        help='The video from which the paths were generated.')
    parser.add_argument('--tracks-files', type=str, nargs='+',
                        help='The file with the saved tracks.')
    parser.add_argument('--out-file', type=str, default='output/dataset',
                        help='The path to the file where the data will be saved')
    parser.add_argument('--append', action='store_true',
                        help='Specify if the data should be added to the out-file (if it exists) or overwritten.')
    parser.add_argument('--filter-moving', action='store_true',
                        help='Specify if you want to automatically filter chunks with large movement.')

    logging.basicConfig(level=logging.DEBUG)
    args = parser.parse_args()

    main(args)
