import gudhi as gd
import sklearn_tda as tda

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn import metrics
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import seaborn as sn

import numpy as np
import pandas as pd
import os
import logging
import itertools

from .chunk_visualiser import ChunkVisualiser


class TDA:

    def __init__(self, chunks, frames, translated_chunks, labels):
        self.chunks = chunks
        self.frames = frames
        self.translated_chunks = translated_chunks
        self.labels = labels
        self.persistences = []

    def visualise_point_clouds(self, data):
        scaler = RobustScaler()
        scaler.fit(data.reshape(-1, 3))
        for i, d in enumerate(data):
            points = scaler.transform(d)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5)
            plt.show()

    def persistence(self, data):
        dim = 3
        betti_numbers = np.zeros((data.shape[0], dim))
        scaler = RobustScaler()
        scaler.fit(data.reshape(-1, 3))

        diags = []
        for i, d in enumerate(data):
            points = scaler.transform(d)

            rips = gd.RipsComplex(max_edge_length=0.5, points=points)
            simplex_tree = rips.create_simplex_tree(max_dimension=3)

            # alpha = gd.AlphaComplex(points=points)
            # simplex_tree = alpha.create_simplex_tree(max_alpha_square=0.1)

            diag_alpha = simplex_tree.persistence()
            # Removing the points who don't die
            clean_diag_alpha = [p for p in diag_alpha if p[1][1] < np.inf]
            self.persistences.append(clean_diag_alpha)
            # tda_diag_df = self._construct_dataframe(clean_diag_alpha)
            # self._betti_curve(tda_diag_df)

            diags.append(np.array([(p[1][1], p[1][0]) for p in clean_diag_alpha]))
            betti = simplex_tree.betti_numbers()
            # Make sure we fill the 2 dimensions
            pad = np.pad(betti, (0, dim - len(betti)), 'constant')
            betti_numbers[i] = pad

        return np.array(diags)

    def _construct_dataframe(self, clean_diag_alpha):
        tda_diag_df = pd.DataFrame()

        tda_diag_df['Dimension'] = [el[0] for el in clean_diag_alpha]
        tda_diag_df['Birth'] = [el[1][0] for el in clean_diag_alpha]
        tda_diag_df['Death'] = [el[1][1] for el in clean_diag_alpha]
        tda_diag_df['Lifespan'] = tda_diag_df['Death'] - tda_diag_df['Birth']
        return tda_diag_df

    def _betti_curve(self, tda_diag_df, dim):
        betti_points = 100

        betti_curve_0 = []
        min_birth = tda_diag_df.loc[tda_diag_df.Dimension == dim].Birth.min()
        max_death = tda_diag_df.loc[tda_diag_df.Dimension == dim].Death.max()
        betti_range = np.linspace(min_birth, max_death, betti_points)
        for death in betti_range:
            nb_points_alive = tda_diag_df.loc[
                (tda_diag_df.Dimension == dim) & (tda_diag_df.Death >= death)].shape[0]
            betti_curve_0.append([death, nb_points_alive])
        betti_curve_0 = np.array(betti_curve_0)
        plt.plot(betti_curve_0[:, 0], betti_curve_0[:, 1])

    def save_persistences(self, out_dir):

        for i, diag in enumerate(self.persistences):
            fig = gd.plot_persistence_diagram(diag)

            label = self.labels[i]
            plt.title(label)

            file_path = os.path.join(out_dir, 'persistence-{}.png'.format(i))
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()

    def save_betti_curves(self, out_dir):
        for i, diag in enumerate(self.persistences):
            tda_diag_df = self._construct_dataframe(diag)

            label = self.labels[i]
            plt.title(label)

            for dim in range(3):
                self._betti_curve(tda_diag_df, dim)

                file_path = os.path.join(out_dir, 'betti-curve-{}-{}.png'.format(i, dim))
                plt.savefig(file_path, bbox_inches='tight')
                plt.close()

    def classify(self, data, labels_true):
        le = LabelEncoder()
        labels_true = le.fit_transform(labels_true)
        # data = RobustScaler().fit_transform(data)

        logging.debug("Splitting data into test/train")
        train_data, test_data, train_labels, test_labels = sklearn.model_selection.train_test_split(
            data, labels_true)

        logging.debug("Cross-validating to find best model.")
        model = self._pipeline()
        model = model.fit(train_data, train_labels)
        # print(model.best_params_)
        # logging.info("Train accuracy = {}".format(model.score(train_data, train_labels)))
        labels = model.predict(test_data)
        test_accuracy = metrics.accuracy_score(test_labels, labels)
        logging.info("Test accuracy: {}".format(test_accuracy))
        self._plot_confusion_matrix(labels, test_labels, le)

        return labels

    def _plot_confusion_matrix(self, labels, test_labels, le):
        confusion_matrix = metrics.confusion_matrix(test_labels, labels).astype(np.float32)
        confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)[:, np.newaxis]
        class_names = list(le.classes_)
        df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)
        plt.show()

    def _pipeline(self):

        # Definition of pipeline# Defin
        pipe = Pipeline([
            ("Separator", tda.DiagramSelector(limit=np.inf, point_type="finite")),
            ("TDA",       tda.SlicedWasserstein(bandwidth=1.0, num_directions=10)),
            ("Estimator", SVC(kernel='precomputed'))
        ])

        return pipe

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

    def visualise(self, labels, videos):
        visualiser = ChunkVisualiser(self.chunks, self.frames, self.translated_chunks)
        unique_labels = set(labels)
        for k in unique_labels:
            if k == -1:
                continue

            class_member_mask = (labels == k)
            node = np.where(class_member_mask)[0]
            name = "Cluster " + str(k)
            visualiser.draw_node(videos, name, node)

    def _add_on_click(self, figs, data, axes):
        for ax in axes:
            ax.plot(data[:, 0], data[:, 1], data[:, 2], 'o', markerfacecolor='b',
                    markeredgecolor='b', markersize=0, picker=5)

        def onpick(event):
            visualiser = ChunkVisualiser(self.chunks, self.frames, self.translated_chunks)
            ind = event.ind
            nodes = {'Picked points': ind}
            visualiser.visualise_averages(nodes)

        for fig in figs:
            fig.canvas.mpl_connect('pick_event', onpick)
