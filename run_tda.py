import argparse
import logging
import os
import numpy as np
import json


from tracker import Person, Track, TrackVisualiser
from analysis import PostProcessor, Mapper, TDA
from util import COCOKeypoints


def main(args):
    dataset_npz = np.load(args.dataset)
    chunks = dataset_npz['chunks']
    frames = dataset_npz['frames']
    labels = dataset_npz['labels']
    videos = dataset_npz['videos']

    processor = PostProcessor()
    logging.info("Normalising every limb to a fixed size.")
    normalised_chunks = processor.normalise_limb_lengths_of_chunks(chunks)
    # logging.info("Translating every chunk by the average position of that chunk.")
    # translated_chunks = processor.translate_chunks_to_origin(normalised_chunks)
    logging.info("Translating every body part by the average position of that body part in the chunk.")
    translated_chunks = processor.translate_chunks_to_origin_by_keypoint(normalised_chunks)

    selected_keypoints = [
        COCOKeypoints.RWrist.value,
        COCOKeypoints.LWrist.value,
        COCOKeypoints.RElbow.value,
        COCOKeypoints.LElbow.value
    ]
    logging.info("Flattening data into a datapoint per chunk of a track.")
    data = processor.flatten_chunks(translated_chunks, frames, selected_keypoints)
    # data = processor.velocity_of_chunks(translated_chunks, frames, selected_keypoints)

    # visualiser = TrackVisualiser()
    # filtered_tracks = processor.chunks_to_tracks(chunks, frames)
    # visualiser.draw_video_with_tracks(filtered_tracks, args.video, last_frame)
    if args.tda:
        run_tda(chunks, frames, videos, labels, data)
    if args.mapper:
        run_mapper(chunks, frames, translated_chunks, videos,
                   labels, data)


def run_tda(chunks, frames, videos, labels, data):
    logging.info("Applying TDA with gudhi to chunks.")
    tda = TDA()
    # betti_numbers = tda.persistence(data)
    tda.cluster(data, labels)


def run_mapper(chunks, frames, translated_chunks, videos, labels, data):
    logging.info("Applying mapping to tracks.")
    mapper = Mapper(chunks, frames, translated_chunks, labels)
    mapper.create_tooltips(videos)
    graph = mapper.mapper(data)
    logging.info("Visualisation of the resulting nodes.")
    mapper.visualise(videos, graph)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TDA analysis of the tracks.')
    parser.add_argument('--dataset', type=str, help='The path to the dataset')
    parser.add_argument('--mapper', action='store_true',
                        help='Run the mapper algorithm on the data')
    parser.add_argument('--tda', action='store_true',
                        help='Run a different TDA algorith on the data.')

    logging.basicConfig(level=logging.DEBUG)
    args = parser.parse_args()

    main(args)
