import argparse
import logging
import os
import numpy as np
import json
import cv2
from collections import Counter

from tracker import Person, Track, TrackVisualiser
from analysis import Mapper, TDAClassifier, ChunkVisualiser
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

    if args.tda:
        run_tda(chunks, frames, translated_chunks, videos, labels)
    if args.mapper:
        run_mapper(chunks, frames, translated_chunks, videos, labels)
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


def run_tda(chunks, frames, translated_chunks, videos, labels):
    classifier = TDAClassifier(chunks, frames, translated_chunks, labels, videos)
    pred_labels, test_labels, le = classifier.classify(chunks, labels)
    classifier.visualise(pred_labels, test_labels, le)


def run_mapper(chunks, frames, translated_chunks, videos, labels):
    logging.info("Smoothing chunks.")
    translated_chunks = SmoothChunks().transform(translated_chunks)

    selected_keypoints = [
        COCOKeypoints.RShoulder.value,
        COCOKeypoints.LShoulder.value,
        COCOKeypoints.RElbow.value,
        COCOKeypoints.LElbow.value,
        COCOKeypoints.RWrist.value,
        COCOKeypoints.LWrist.value
    ]
    logging.info("Flattening data into a datapoint per chunk.")
    data = Flatten(selected_keypoints).transform(translated_chunks)
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
                        help='Run a TDA algorithm on the data.')
    parser.add_argument('--visualise', action='store_true',
                        help='Specify if you wish to only visualise the classes in the dataset.')

    logging.basicConfig(level=logging.DEBUG)
    args = parser.parse_args()

    main(args)
