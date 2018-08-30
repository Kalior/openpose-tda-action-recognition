import argparse
import numpy as np

from action_recognition.util import load_data


def main(args):
    datasets = args.datasets
    train_name = datasets[0] + '-train.npz'
    train_chunks, train_frames, train_labels, train_videos = load_data(train_name)
    test_name = datasets[0] + '-test.npz'
    test_chunks, test_frames, test_labels, test_videos = load_data(test_name)

    for i in range(1, len(datasets)):
        train_name = datasets[i] + '-train.npz'
        new_train_chunks, new_train_frames, new_train_labels, new_train_videos = load_data(
            train_name)
        train_chunks = np.append(train_chunks, new_train_chunks, axis=0)
        train_frames = np.append(train_frames, new_train_frames, axis=0)
        train_labels = np.append(train_labels, new_train_labels, axis=0)
        train_videos = np.append(train_videos, new_train_videos, axis=0)

        test_name = datasets[i] + '-test.npz'
        new_test_chunks, new_test_frames, new_test_labels, new_test_videos = load_data(test_name)
        test_chunks = np.append(test_chunks, new_test_chunks, axis=0)
        test_frames = np.append(test_frames, new_test_frames, axis=0)
        test_labels = np.append(test_labels, new_test_labels, axis=0)
        test_videos = np.append(test_videos, new_test_videos, axis=0)

    train_chunks, train_frames, train_labels, train_videos = shuffle(
        train_chunks, train_frames, train_labels, train_videos)
    test_chunks, test_frames, test_labels, test_videos = shuffle(
        test_chunks, test_frames, test_labels, test_videos)

    train_name = args.out_file + '-train.npz'
    test_name = args.out_file + '-test.npz'
    np.savez(train_name, chunks=train_chunks, frames=train_frames,
             labels=train_labels, videos=train_videos)
    np.savez(test_name, chunks=test_chunks, frames=test_frames,
             labels=test_labels, videos=test_videos)


def shuffle(chunks, frames, labels, videos):
    permutation = np.random.permutation(chunks.shape[0])
    chunks = chunks[permutation]
    frames = frames[permutation]
    labels = labels[permutation]
    videos = videos[permutation]
    return chunks, frames, labels, videos


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('Combines mutliple dataset files into one. '
                     'Keeps the train and test splits from the individual sets, so '
                     'it does not re-split into test and train.  It does, however, '
                     'shuffle the existing splits to avoid bias towards specific '
                     'features of a single set.'))
    parser.add_argument('--datasets', type=str, nargs='+', help='The datasets to combine.')
    parser.add_argument('--out-file', type=str, help='Name of the new dataset.')

    args = parser.parse_args()

    main(args)
