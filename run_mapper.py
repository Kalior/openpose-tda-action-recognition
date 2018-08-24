import argparse
import logging
import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from action_recognition.analysis import Mapper
from action_recognition import transforms
from action_recognition.util import COCOKeypoints


def main(args):
    extension_free = os.path.splitext(args.dataset)[0]
    train_name = extension_free + '-train.npz'
    test_name = extension_free + '-test.npz'
    train = load_data(train_name)
    test = load_data(test_name)

    logging.info("Number of train dataset labels: {}".format(Counter(train[2])))
    logging.info("Number of test dataset labels: {}".format(Counter(test[2])))

    run_mapper(train, test)


def load_data(file_name):
    dataset_npz = np.load(file_name)
    # Converts the data into a non-object array if possible
    chunks = np.array([t for t in dataset_npz['chunks']])
    frames = dataset_npz['frames']
    labels = dataset_npz['labels']
    videos = dataset_npz['videos']

    return chunks, frames, labels, videos


def append_train_and_test(train, test):
    chunks = np.append(train[0], test[0], axis=0)
    frames = np.append(train[1], test[1], axis=0)
    labels = np.append(train[2], test[2], axis=0)
    videos = np.append(train[3], test[3], axis=0)
    return chunks, frames, labels, videos


def run_mapper(test, train):
    chunks, frames, labels, videos = append_train_and_test(train, test)
    translated_chunks = transforms.TranslateChunks().transform(chunks)

    logging.info("Smoothing chunks.")
    translated_chunks = transforms.SmoothChunks().transform(translated_chunks)

    selected_keypoints = [
        COCOKeypoints.RShoulder.value,
        COCOKeypoints.LShoulder.value,
        COCOKeypoints.RElbow.value,
        COCOKeypoints.LElbow.value,
        COCOKeypoints.RWrist.value,
        COCOKeypoints.LWrist.value
    ]
    logging.info("Flattening data into a datapoint per chunk.")
    extracted = transforms.ExtractKeypoints(selected_keypoints).transform(translated_chunks)
    data = transforms.Flatten().transform(extracted)
    logging.info("Applying mapping to tracks.")
    mapper = Mapper(chunks, frames, translated_chunks, labels)
    mapper.create_tooltips(videos)
    graph = mapper.mapper(data)
    logging.info("Visualisation of the resulting nodes.")
    mapper.visualise(videos, graph)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('Run the mapper algorithm on the dataset.  Can help visualise the data. '
                     'Requires that the data has equal dimensions, i.e. the same number of frames per chunk.'))
    parser.add_argument('--dataset', type=str, help='The path to the dataset')

    logging.basicConfig(level=logging.DEBUG)
    args = parser.parse_args()

    main(args)
