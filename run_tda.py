import argparse
import logging
import os
import numpy as np
import json


from tracker import Person, Track, TrackVisualiser
from analysis import PostProcessor, Mapper, TDA, Labelling
from util import COCOKeypoints


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
    translated_chunks = processor.translate_chunks_to_origin(static_chunks)

    labels_file = os.path.splitext(args.tracks_file)[0] + '.labels'
    if os.path.isfile(labels_file):
        with open(labels_file, 'r') as f:
            labels = json.load(f)
    else:
        labels = Labelling.label_chunks(static_chunks, static_frames, args.video, processor)
        j = json.dump(labels)
        with open(labels_file, 'w') as f:
            f.write(j)

    logging.info("Flattening data into a datapoint per chunk of a track.")
    selected_keypoints = [
        COCOKeypoints.RWrist.value,
        COCOKeypoints.LWrist.value,
        COCOKeypoints.RElbow.value,
        COCOKeypoints.LElbow.value
    ]
    data, meta = processor.flatten_chunks(translated_chunks, static_frames, selected_keypoints)
    # data, labels = processor.velocity_of_chunks(translated_chunks, static_frames, selected_keypoints)

    # visualiser = TrackVisualiser()
    # filtered_tracks = processor.chunks_to_tracks(static_chunks, static_frames)
    # visualiser.draw_video_with_tracks(filtered_tracks, args.video, last_frame)
    if args.tda:
        run_tda(static_chunks, static_frames, frames_per_chunk, args.video, labels, data, meta)
    if args.mapper:
        run_mapper(static_chunks, static_frames, translated_chunks, frames_per_chunk, args.video,
                   labels, data, meta)


def run_tda(chunks, frames, frames_per_chunk, video, labels, data, meta):
    logging.info("Applying TDA with gudhi to chunks.")
    tda = TDA()


def run_mapper(chunks, frames, translated_chunks, frames_per_chunk, video, labels, data, meta):
    logging.info("Applying mapping to tracks.")
    mapper = Mapper(chunks, frames, translated_chunks, frames_per_chunk, labels)
    mapper.create_tooltips(video)
    graph = mapper.mapper(data)
    logging.info("Visualisation of the resulting nodes.")
    mapper.visualise(video, graph, meta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualisation of tracking system.')
    parser.add_argument('--video', type=str,
                        help='The video from which the paths were generated.')
    parser.add_argument('--tracks-file', type=str, default='output/paths/',
                        help='The file with the saved tracks.')
    parser.add_argument('--mapper', action='store_true',
                        help='Run the mapper algorithm on the data')
    parser.add_argument('--tda', action='store_true',
                        help='Run a different TDA algorith on the data.')

    logging.basicConfig(level=logging.DEBUG)
    args = parser.parse_args()

    main(args)
