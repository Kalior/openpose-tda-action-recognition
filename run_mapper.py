import argparse
import logging
import os
import numpy as np


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

    # visualiser = TrackVisualiser()
    # filtered_tracks = processor.chunks_to_tracks(static_chunks, static_frames)
    # visualiser.draw_video_with_tracks(filtered_tracks, args.video, last_frame)

    logging.info("Applying mapping to tracks.")
    mapper = Mapper(static_chunks, static_frames, frames_per_chunk, args.video)
    graph, _, labels = mapper.mapper()
    mapper.visualise(graph, labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualisation of tracking system.')
    parser.add_argument('--video', type=str,
                        help='The video from which the paths were generated.')
    parser.add_argument('--tracks-file', type=str, default='output/paths/',
                        help='The file with the saved tracks.')

    logging.basicConfig(level=logging.DEBUG)
    args = parser.parse_args()

    main(args)
