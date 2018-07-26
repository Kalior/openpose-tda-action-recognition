import gudhi.gudhi as gudhi
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler
from sklearn import metrics
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import manifold

import numpy as np
import pandas as pd

from .chunk_visualiser import ChunkVisualiser


class TDA:

    def __init__(self, chunks, frames, translated_chunks, labels):
        self.chunks = chunks
        self.frames = frames
        self.translated_chunks = translated_chunks
        self.labels = labels

    def persistence(self, data):
        dim = 3
        betti_numbers = np.zeros((data.shape[0], dim))
        scaler = RobustScaler()
        scaler.fit(data.reshape(-1, 3))
        for i, d in enumerate(data):
            points = scaler.transform(d)

            if i < 5:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5)
                plt.show()

            rips = gudhi.RipsComplex(max_edge_length=0.5, points=points)
            simplex_tree = rips.create_simplex_tree(max_dimension=3)

            # alpha = gudhi.AlphaComplex(points=points)
            # simplex_tree = alpha.create_simplex_tree(max_alpha_square=0.1)

            diag_alpha = simplex_tree.persistence()
            # Removing the points who don't die
            clean_diag_alpha = [p for p in diag_alpha if p[1][1] < np.inf]
            tda_diag_df = self._construct_dataframe(clean_diag_alpha)
            # self._betti_curve(tda_diag_df)

            # gudhi.plot_persistence_diagram(diag_alpha)
            # plt.show()

            betti = simplex_tree.betti_numbers()
            # Make sure we fill the 2 dimensions
            pad = np.pad(betti, (0, dim - len(betti)), 'constant')
            betti_numbers[i] = pad

        return betti_numbers

    def _construct_dataframe(self, clean_diag_alpha):
        tda_diag_df = pd.DataFrame()

        tda_diag_df['Dimension'] = [el[0] for el in clean_diag_alpha]
        tda_diag_df['Birth'] = [el[1][0] for el in clean_diag_alpha]
        tda_diag_df['Death'] = [el[1][1] for el in clean_diag_alpha]
        tda_diag_df['Lifespan'] = tda_diag_df['Death'] - tda_diag_df['Birth']
        return tda_diag_df

    def _betti_curve(self, tda_diag_df):
        betti_points = 100
        dim = 0

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
        plt.show()

    def cluster(self, data, labels_true):
        le = preprocessing.LabelEncoder()
        labels_true = le.fit_transform(labels_true)

        db = DBSCAN(eps=0.3, min_samples=2).fit(data)

        labels = db.labels_
        self._print_cluster_metrics(labels_true, labels, data)

        # Set up a figure twice as wide as it is tall
        true_fig = plt.figure(figsize=plt.figaspect(0.5))
        true_ax = true_fig.add_subplot(1, 1, 1, projection='3d')
        cluster_fig = plt.figure(figsize=plt.figaspect(0.5))
        cluster_ax = cluster_fig.add_subplot(1, 1, 1, projection='3d')

        # Perturb the datapoints so we see multiple points in the visualisation
        data = data + np.random.normal(loc=0, scale=0.1, size=data.shape)

        self._plot_clusters(data, labels_true, "True", le, true_ax)
        self._plot_clusters(data, labels, "Estimated", le, cluster_ax)

        self._add_on_click([true_fig, cluster_fig], data, [true_ax, cluster_ax])
        plt.show()

        return labels

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
