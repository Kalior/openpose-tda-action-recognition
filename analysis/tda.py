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

import numpy as np
import pandas as pd
import os
import logging

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

    def cluster(self, data, labels_true):
        le = LabelEncoder()
        labels_true = le.fit_transform(labels_true)
        # data = RobustScaler().fit_transform(data)

        logging.info("Shuffling the data")
        p = np.random.permutation(len(data))
        data = data[p]
        labels_true = labels_true[p]

        test_split = int(len(data) / 3)
        train_data = data[test_split:]
        train_labels = labels_true[test_split:]
        test_data = data[:test_split]
        test_labels = labels_true[:test_split]

        logging.info("Cross-validating to find best model.")
        model = self._cross_validate_pipeline()
        model = model.fit(train_data, train_labels)
        print(model.best_params_)
        print("Train accuracy = " + str(model.score(train_data, train_labels)))
        print("Test accuracy  = " + str(model.score(test_data,  test_labels)))
        labels = model.predict(test_data)
        transform_data = model.transform(test_data)

        # self._print_cluster_metrics(labels_true, labels, data)

        # Set up a figure twice as wide as it is tall
        true_fig = plt.figure(figsize=plt.figaspect(0.5))
        true_ax = true_fig.add_subplot(1, 1, 1, projection='3d')
        cluster_fig = plt.figure(figsize=plt.figaspect(0.5))
        cluster_ax = cluster_fig.add_subplot(1, 1, 1, projection='3d')

        # Perturb the datapoints so we see multiple points in the visualisation
        # data = data + np.random.normal(loc=0, scale=0.1, size=data.shape)

        self._plot_clusters(transform_data, test_labels, "True", le, true_ax)
        self._plot_clusters(transform_data, labels, "Estimated", le, cluster_ax)

        self._add_on_click([true_fig, cluster_fig], data, [true_ax, cluster_ax])
        plt.show()

        return labels

    def _cross_validate_pipeline(self):

        # Definition of pipeline# Defin
        pipe = Pipeline([
            ("Separator", tda.DiagramSelector(limit=np.inf, point_type="finite")),
            ("Rotator",   tda.DiagramPreprocessor(
                scaler=tda.BirthPersistenceTransform())),
            ("TDA",       tda.PersistenceImage()),
            ("Estimator", SVC())
        ])

        # Parameters of pipeline. This is the place where you specify the methods
        # you want to use to handle diagrams
        param = [
            {
                "Rotator__use":        [False],
                "TDA":                 [tda.SlicedWasserstein()],
                "TDA__bandwidth":      [0.1, 1.0],
                "TDA__num_directions": [20],
                "Estimator":           [SVC(kernel="precomputed")]
            },
            {
                "Rotator__use":        [False],
                "TDA":                 [tda.PersistenceWeightedGaussian()],
                "TDA__bandwidth":      [0.1, 1.0],
                "TDA__weight":         [lambda x: np.arctan(x[1] - x[0])],
                "Estimator":           [SVC(kernel="precomputed")]
            },

            {
                "Rotator__use":        [True],
                "TDA":                 [tda.PersistenceImage()],
                "TDA__resolution":     [[5, 5], [6, 6]],
                "TDA__bandwidth":      [0.01, 0.1, 1.0, 10.0],
                "Estimator":           [SVC()]
            },
            {
                "Rotator__use":        [False],
                "TDA":                 [tda.Landscape()],
                "TDA__resolution":     [1000],
                "Estimator":           [RandomForestClassifier()]
            }
        ]
        return sklearn.model_selection.GridSearchCV(pipe, param, cv=3)

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

    def _print_cluster_metrics(self, labels_true, labels, data):
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        print('Estimated number of clusters: {}'.format(n_clusters_))
        print("Homogeneity: {:.3f}".format(metrics.homogeneity_score(labels_true, labels)))
        print("Completeness: {:.3f}".format(metrics.completeness_score(labels_true, labels)))
        print("V-measure: {:.3f}".format(metrics.v_measure_score(labels_true, labels)))
        print("Adjusted Rand Index: {:.3f}".format(
            metrics.adjusted_rand_score(labels_true, labels)))
        print("Adjusted Mutual Information: {:.3f}".format(
            metrics.adjusted_mutual_info_score(labels_true, labels)))
        print("Silhouette Coefficient: {:.3f}".format(
              metrics.silhouette_score(data, labels)))

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
