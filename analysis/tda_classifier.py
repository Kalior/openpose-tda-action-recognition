import sklearn_tda as tda

import matplotlib.pyplot as plt

import sklearn
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

import seaborn as sn

import numpy as np
import pandas as pd
import itertools
import logging

from .persistence import Persistence
from .chunk_visualiser import ChunkVisualiser

from util import COCOKeypoints, coco_connections
from transforms import TranslateChunks, SmoothChunks, FlattenTo3D


class TDAClassifier:

    def __init__(self, chunks, frames, translated_chunks, labels, videos):
        self.chunks = chunks
        self.frames = frames
        self.translated_chunks = translated_chunks
        self.labels = labels
        self.videos = videos

    def classify(self, data, labels_true):
        le = LabelEncoder()
        labels_true = le.fit_transform(labels_true)

        logging.debug("Splitting data into test/train")
        train_data, test_data, \
            train_labels, test_labels, \
            _, test_frames, \
            _, test_videos, \
            _, test_chunks, \
            _, test_translated_chunks = train_test_split(
                data, labels_true, self.frames, self.videos, self.chunks, self.translated_chunks)

        self.test_translated_chunks = test_translated_chunks
        self.test_videos = test_videos
        self.test_chunks = test_chunks
        self.test_frames = test_frames

        logging.debug("Cross-validating to find best model.")
        model = self._pipeline()
        model = model.fit(train_data, train_labels)
        print(model.best_params_)
        # logging.info("Train accuracy = {}".format(model.score(train_data, train_labels)))
        labels = model.predict(test_data)
        test_accuracy = metrics.accuracy_score(test_labels, labels)
        logging.info("Test accuracy: {}".format(test_accuracy))
        self._plot_confusion_matrix(labels, test_labels, le)

        return labels, test_labels, le

    def _plot_confusion_matrix(self, labels, test_labels, le):
        confusion_matrix = metrics.confusion_matrix(test_labels, labels).astype(np.float32)
        confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)[:, np.newaxis]
        class_names = list(le.classes_)
        df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)
        plt.show(block=False)

    def _pipeline(self):
        arm_keypoints = [
            COCOKeypoints.RShoulder.value,
            COCOKeypoints.LShoulder.value,
            COCOKeypoints.RElbow.value,
            COCOKeypoints.LElbow.value,
            COCOKeypoints.RWrist.value,
            COCOKeypoints.LWrist.value
        ]
        arm_connections = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5), (4, 5)]
        all_keypoints = range(18)

        # Definition of pipeline
        pipe = Pipeline([
            ("Translate", TranslateChunks()),
            ("Smoothing", SmoothChunks()),
            ("Flattening", FlattenTo3D(arm_keypoints, arm_connections, True)),
            ("Persistence", Persistence()),
            ("Separator", tda.DiagramSelector(limit=np.inf, point_type="finite")),
            ("TDA",       tda.SlicedWasserstein(bandwidth=1.0, num_directions=10)),
            ("Estimator", SVC(kernel='precomputed'))
        ])

        params = [
            {
                "Smoothing": [None, SmoothChunks()],
                "Flattening__interpolate_points": [True, False],
                "Flattening__selected_keypoints": [arm_keypoints],
                "Flattening__connect_keypoints": [arm_connections]
            },
            {
                "Smoothing": [None, SmoothChunks()],
                "Flattening__interpolate_points": [False],
                "Flattening__selected_keypoints": [arm_keypoints, all_keypoints]
            },
            {
                "Smoothing": [None, SmoothChunks()],
                "Flattening__interpolate_points": [True],
                "Flattening__selected_keypoints": [all_keypoints],
                "Flattening__connect_keypoints": [coco_connections],
            }
        ]

        return GridSearchCV(pipe, params, n_jobs=2)

    def _plot_clusters(self, data, labels, title, le, ax):
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]

        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            if title is "True":
                class_name = le.classes_[k]
            else:
                class_name = k

            xy = data[class_member_mask]
            ax.plot(xy[:, 0], xy[:, 1], xy[:, 2], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=6, label=class_name)

        ax.legend()
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        ax.set_title('{} number of clusters: {}'.format(title, n_clusters_))

    def visualise(self, pred_labels, test_labels, le):
        visualiser = ChunkVisualiser(self.test_chunks,
                                     self.test_frames,
                                     self.test_translated_chunks)
        unique_labels = set(pred_labels)
        for k1 in unique_labels:
            for k2 in unique_labels:
                if k1 == -1 or k2 == -1:
                    continue

                pred_class_member_mask = (pred_labels == k1)
                true_class_member_mask = (test_labels == k2)
                node = np.where(pred_class_member_mask & true_class_member_mask)[0]
                name = "Pred {}\n True {}".format(le.classes_[k1], le.classes_[k2])

                if len(node) != 0:
                    visualiser.draw_node(self.test_videos, name, node)

    def _add_on_click(self, figs, data, axes):
        for ax in axes:
            ax.plot(data[:, 0], data[:, 1], data[:, 2], 'o', markerfacecolor='b',
                    markeredgecolor='b', markersize=0, picker=5)

        def onpick(event):
            visualiser = ChunkVisualiser(
                self.chunks, self.frames, self.translated_chunks)
            ind = event.ind
            nodes = {'Picked points': ind}
            visualiser.visualise_averages(nodes)

        for fig in figs:
            fig.canvas.mpl_connect('pick_event', onpick)
