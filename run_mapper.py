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
    logging.info("Moving every person to the origin.")
    # processor.translate_tracks_to_origin()
    translated_chunks = processor.translate_chunks_to_origin(chunks)
    # print(chunks)

    # visualiser = TrackVisualiser()
    # visualiser.draw_video_with_tracks(processor.tracks, args.video, last_frame)

    logging.info("Applying mapping to tracks.")
    mapper = Mapper(translated_chunks[[2, 3]], chunk_frames[
                    [2, 3]], chunks[[2, 3]], frames_per_chunk)
    mapper.mapper()
    mapper.visualise(args.video)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualisation of tracking system.')
    parser.add_argument('--video', type=str,
                        help='The video from which the paths were generated.')
    parser.add_argument('--tracks-file', type=str, default='output/paths/',
                        help='The file with the saved tracks.')

    logging.basicConfig(level=logging.DEBUG)
    args = parser.parse_args()

    main(args)
