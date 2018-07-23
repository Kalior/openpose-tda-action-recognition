import gudhi.gudhi as gudhi
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import preprocessing
from sklearn import decomposition


import numpy as np


class TDA:


    def cluster(self, data, labels_true):
        le = preprocessing.LabelEncoder()
        labels_true = le.fit_transform(list(labels_true.values()))

        pca = decomposition.PCA(n_components=3)
        X = pca.fit_transform(data)
        X = StandardScaler().fit_transform(X)

        db = DBSCAN(eps=0.5, min_samples=3).fit(X)

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
              metrics.silhouette_score(X, labels)))

        # set up a figure twice as wide as it is tall
        fig = plt.figure(figsize=plt.figaspect(0.5))

        true_ax = fig.add_subplot(1, 2, 1, projection='3d')
        cluster_ax = fig.add_subplot(1, 2, 2, projection='3d')

        self._plot_clusters(X, labels_true, core_samples_mask, "True", le, true_ax)
        self._plot_clusters(X, labels, core_samples_mask, "Estimated", le, cluster_ax)
        plt.show()

    def _plot_clusters(self, X, labels, core_samples_mask, title, le, ax):
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

            xy = X[class_member_mask & core_samples_mask]
            ax.plot(xy[:, 0], xy[:, 1], xy[:, 2], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=14, label=class_name)

            xy = X[class_member_mask & ~core_samples_mask]
            ax.plot(xy[:, 0], xy[:, 1], xy[:, 2], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=6)
        if title is "True":
            ax.legend()
        ax.set_title('{} number of clusters: {}'.format(title, n_clusters_))
