import argparse
import logging
import os
import numpy as np
import json


from tracker import Person, Track, TrackVisualiser
from analysis import PostProcessor, Mapper


def main(args):
    tracks_npz = np.load(args.tracks_file)
    np_tracks = tracks_npz['tracks']
    np_frames = tracks_npz['frames']
    chunks = tracks_npz['chunks']

    last_frame = np.amax([f[-1] for f in np_frames])

    logging.info("Combining, cleaning and removing tracks.")
    processor = PostProcessor()
    processor.create_tracks(np_tracks, np_frames)
    processor.post_process_tracks()

    logging.info("Chunking tracks.")
    frames_per_chunk = 20
    overlap = 10
    chunks, chunk_frames = processor.chunk_tracks(frames_per_chunk, overlap)
    logging.info("Filtering out every path but the cachier standing still.")
    static_chunks, static_frames = processor.filter_moving_chunks(chunks, chunk_frames)

    labels_file = os.path.splitext(args.tracks_file)[0] + '.labels'
    if os.path.isfile(labels_file):
        with open(labels_file, 'r') as f:
            labels = json.load(f)
    else:
        labels = label_chunks(static_chunks, static_frames, args.video, processor)
        j = json.dump(labels)
        with open(labels_file, 'w') as f:
            f.write(j)

    # visualiser = TrackVisualiser()
    # filtered_tracks = processor.chunks_to_tracks(static_chunks, static_frames)
    # visualiser.draw_video_with_tracks(filtered_tracks, args.video, last_frame)

    logging.info("Applying mapping to tracks.")
    mapper = Mapper(static_chunks, static_frames, frames_per_chunk, args.video, labels)
    graph, data_labels = mapper.mapper()
    logging.info("Visualisation of the resulting nodes.")
    mapper.visualise(graph, data_labels)


def label_chunks(chunks, chunk_frames, video, processor):
    visualiser = TrackVisualiser()
    tracks = processor.chunks_to_tracks(chunks, chunk_frames)

    labels = {}

    for i, track in enumerate(tracks):
        visualiser.draw_video_with_tracks(
            [track], video, track.frame_assigned[-1].astype(np.int), track.frame_assigned[0].astype(np.int))
        label = input('Label? (Scan, Cash, sTill, Moving, Other)')
        if label in ['s', 'c', 'o', 'm', 't']:
            if label == 's':
                label = 'scan'
            elif label == 'c':
                label = 'cash'
            elif label == 'm':
                label = 'moving'
            elif label == 't':
                label = 'still'
            else:
                label = 'other'
        else:
            label = 'other'
        labels[i] = label
        print()

    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualisation of tracking system.')
    parser.add_argument('--video', type=str,
                        help='The video from which the paths were generated.')
    parser.add_argument('--tracks-file', type=str, default='output/paths/',
                        help='The file with the saved tracks.')

    logging.basicConfig(level=logging.DEBUG)
    args = parser.parse_args()

    main(args)
