import argparse
import logging
import os
import numpy as np
import json
import cv2
from collections import Counter

from tracker import Person, Track, TrackVisualiser
from analysis import Mapper, TDA, ChunkVisualiser
from transforms import Flatten, FlattenTo3D, SmoothChunks, TranslateChunks, TranslateChunksByKeypoints
from util import COCOKeypoints


def main(args):
    dataset_npz = np.load(args.dataset)
    chunks = dataset_npz['chunks']
    frames = dataset_npz['frames'].astype(np.int)
    labels = dataset_npz['labels']
    videos = dataset_npz['videos']

    logging.info("Number of dataset labels: {}".format(Counter(labels)))

    logging.info("Translating every chunk by the average position of that chunk.")
    translated_chunks = TranslateChunks().transform(chunks)
    # logging.info("Translating every body part by the average position of that body part in the chunk.")
    # translated_chunks = TranslateChunksByKeypoints().transform(chunks)
    logging.info("Smoothing the path of the keypoints.")
    translated_chunks = SmoothChunks().transform(translated_chunks)

    selected_keypoints = [
        COCOKeypoints.RShoulder.value,
        COCOKeypoints.LShoulder.value,
        COCOKeypoints.RElbow.value,
        COCOKeypoints.LElbow.value,
        COCOKeypoints.RWrist.value,
        COCOKeypoints.LWrist.value
    ]
    connect_keypoints = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5)]

    if args.tda:
        logging.info("Flattening data into 3D, with third dimension as time.")
        data = FlattenTo3D(selected_keypoints, connect_keypoints).transform(translated_chunks)
        run_tda(chunks, frames, translated_chunks, videos, labels, data)
    if args.mapper:
        logging.info("Flattening data into a datapoint per chunk.")
        data = Flatten(selected_keypoints).transform(translated_chunks)
        run_mapper(chunks, frames, translated_chunks, videos, labels, data)
    if args.visualise:
        visualise_classes(chunks, frames, translated_chunks, labels)


def visualise_classes(chunks, frames, translated_chunks, labels):
    visualiser = ChunkVisualiser(chunks, frames, translated_chunks)
    unique_labels = set(labels)
    nodes = {}
    for k in unique_labels:
        if k == -1:
            continue

        class_member_mask = (labels == k)
        node = np.where(class_member_mask)[0]
        name = str(k)
        nodes[name] = node

    visualiser.visualise_averages(nodes, True)




def run_tda(chunks, frames, translated_chunks, videos, labels, data):
    logging.info("Applying TDA with gudhi to chunks.")
    tda = TDA(chunks, frames, translated_chunks, labels)
    betti_numbers = tda.persistence(data)
    logging.info("Clustering the betti numbers.")
    cluster_labels = tda.cluster(betti_numbers, labels)
    # tda.visualise(cluster_labels, videos)


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
    parser.add_argument('--visualise', action='store_true',
                        help='Specify if you wish to only visualise the classes in the dataset.')

    logging.basicConfig(level=logging.DEBUG)
    args = parser.parse_args()

    main(args)
