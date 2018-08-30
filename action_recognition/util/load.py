import numpy as np


def load_data(file_name):
    dataset_npz = np.load(file_name)
    # Converts the data into a non-object array if possible
    chunks = np.array([t for t in dataset_npz['chunks']])
    frames = dataset_npz['frames']
    labels = dataset_npz['labels']
    videos = dataset_npz['videos']

    return chunks, frames, labels, videos
