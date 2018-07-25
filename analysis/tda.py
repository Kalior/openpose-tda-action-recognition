import gudhi.gudhi as gudhi
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import manifold

import numpy as np
import pandas as pd

from .chunk_visualiser import ChunkVisualiser


class TDA:

    def persistence(self, data):
        dim = 2
        betti_numbers = np.empty((data.shape[0], dim))
        for i, d in enumerate(data):
            points = d.reshape(-1, 2)
            points = StandardScaler().fit_transform(points)
            # plt.plot(points[:, 0], points[:, 1], 'bo')
            # plt.show()
            rips = gudhi.RipsComplex(max_edge_length=0.8,
                                     points=points)

            simplex_tree = rips.create_simplex_tree(max_dimension=2)

            diag_alpha = simplex_tree.persistence(homology_coeff_field=11)
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

        print(betti_numbers)
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

        db = DBSCAN(eps=0.8, min_samples=10).fit(data)

        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

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

        # Set up a figure twice as wide as it is tall
        fig, [true_ax, cluster_ax] = plt.subplots(1, 2, figsize=plt.figaspect(0.5))

        # Perturb the datapoints so we see multiple points in the visualisation
        data = data + np.random.normal(loc=0, scale=0.1, size=data.shape)

        self._plot_clusters(data, labels_true, core_samples_mask, "True", le, true_ax)
        self._plot_clusters(data, labels, core_samples_mask, "Estimated", le, cluster_ax)

        self._add_on_click(fig, data, true_ax, cluster_ax)
        plt.show()

        return labels

    def _plot_clusters(self, data, labels, core_samples_mask, title, le, ax):
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

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

            xy = data[class_member_mask & core_samples_mask]
            ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=14, label=class_name)

            xy = data[class_member_mask & ~core_samples_mask]
            ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=6)

        ax.legend()
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

    def _add_on_click(self, fig, data, true_ax, cluster_ax):
        for ax in [true_ax, cluster_ax]:
            ax.plot(data[:, 0], data[:, 1], 'o', markerfacecolor='b',
                    markeredgecolor='b', markersize=0, picker=5)

        def onpick(event):
            visualiser = ChunkVisualiser(self.chunks, self.frames, self.translated_chunks)
            ind = event.ind
            print(ind)
            nodes = {'Picked points': ind}
            visualiser.visualise_averages(nodes)

        fig.canvas.mpl_connect('pick_event', onpick)
