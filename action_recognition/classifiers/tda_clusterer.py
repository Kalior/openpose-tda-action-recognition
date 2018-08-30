import matplotlib.pyplot as plt
import sklearn
import sklearn_tda as tda
import numpy as np
import logging

from ..transforms import TranslateChunks, SmoothChunks, ExtractKeypoints, InterpolateKeypoints, \
    FlattenTo3D, Persistence
from ..util import COCOKeypoints


class TDAClusterer:

    def fit_predict(self, X):
        arm_keypoints = [
            COCOKeypoints.RElbow.value,
            COCOKeypoints.RWrist.value,
            COCOKeypoints.LElbow.value,
            COCOKeypoints.LWrist.value
        ]
        arm_connections = [(0, 1), (2, 3)]

        logging.info("Translate.")
        X = TranslateChunks().transform(X)
        logging.info("Extract.")
        X = ExtractKeypoints(arm_keypoints).transform(X)
        logging.info("Smooth.")
        X = SmoothChunks().transform(X)
        logging.info("Interpolate.")
        X = InterpolateKeypoints(arm_connections).transform(X)
        logging.info("Flatten.")
        X = FlattenTo3D().transform(X)
        logging.info("Persistence.")
        X = Persistence().fit_transform(X)
        logging.info("Diagram selector.")
        X = tda.DiagramSelector(limit=np.inf, point_type="finite").fit_transform(X)
        logging.info("Sliced wasserstein.")
        X = tda.SlicedWasserstein(bandwidth=0.6, num_directions=20).fit_transform(X)

        logging.info("Cluster.")
        clusterer = sklearn.cluster.DBSCAN(metric='precomputed')
        # clusterer = sklearn.cluster.SpectralClustering(n_clusters=5, affinity='precomputed')
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
