import matplotlib.pyplot as plt
import sklearn
import sklearn_tda as tda
import numpy as np
import logging

from .. import transforms
from ..util import COCOKeypoints


class TDAClusterer:

    def fit_predict(self, X):
        selected_keypoints = [
            COCOKeypoints.Neck.value,
            COCOKeypoints.RWrist.value,
            COCOKeypoints.LWrist.value,
            COCOKeypoints.RAnkle.value,
            COCOKeypoints.LAnkle.value,
        ]

        pipe = sklearn.pipeline.Pipeline([
            ("Extract", transforms.ExtractKeypoints(selected_keypoints)),
            ("Smoothing", transforms.SmoothChunks()),
            ("Translate", transforms.TranslateChunks()),
            ("PositionCloud", transforms.FlattenTo3D()),
            ("Persistence", transforms.Persistence(max_alpha_square=2, complex_='alpha')),
            ("Separator", tda.DiagramSelector(limit=np.inf, point_type="finite")),
            ("Prominent", tda.ProminentPoints()),
            ("TDA", tda.SlicedWasserstein(bandwidth=0.6, num_directions=20))
        ])

        X = pipe.fit_transform(X)

        logging.info("Cluster.")
        clusterer = sklearn.cluster.DBSCAN(metric='precomputed')
        y_pred = clusterer.fit_predict(X)

        logging.info("PCA.")
        pca = sklearn.decomposition.KernelPCA(n_components=3, kernel='precomputed')
        x_transformed = pca.fit_transform(X)

        return y_pred, x_transformed

    def plot_clusters(self, X, labels, title):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = X[class_member_mask]
            ax.plot(xy[:, 0], xy[:, 1], xy[:, 2], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=6, label=k)

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        plt.title('{} number of clusters: {}'.format(title, n_clusters_))
        plt.legend()
        plt.show(block=False)
