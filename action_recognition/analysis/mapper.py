import kmapper as km
import sklearn
from sklearn import ensemble
import numpy as np
import os
import logging

from .chunk_visualiser import ChunkVisualiser
from .post_processor import PostProcessor


class Mapper:
    """Runs the mapper algorithm on the dataset.

    Essentially a visualisation procedure for high-dimensional data.
    https://github.com/MLWave/kepler-mapper/

    Parameters
    ----------
    chunks : array-like, shape = [n_chunks, n_frames, n_keypoints, 2]
        The chunks of the input data, used only for visualisation.
    chunk_frames : array-like, shape = [n_chunks, n_frames, 1]
        The corresponding frames to the chunks, used only for visualisation.
    translated_chunks : array-like, shape = [n_chunks, n_frames, n_keypoints, 2]
        The chunks translated to origin, used for visualisation.
    labels : array-like, shape = [n_chunks, 1]
        Labels of the chunks, used for visualisation.

    """

    def __init__(self, chunks, chunk_frames, translated_chunks, labels):
        self.chunks = chunks
        self.chunk_frames = chunk_frames
        self.translated_chunks = translated_chunks
        self.chunk_visualiser = ChunkVisualiser(chunks, chunk_frames, translated_chunks)
        self.labels = labels
        self.tooltips = labels

    def visualise(self, videos, graph):
        """Visualises the data by displaying the videos corresponding to the clustering of chunks.

        Parameters
        ----------
        videos : array-like, shape = [n_chunks, 1]
            the videos of the chunks.
        graph : dict
            The graph output from km.KeplerMapper(...).map(...)

        """
        self.chunk_visualiser.visualise(videos, graph)

    def mapper(self, data):
        """Run the mapper algorithm on the data.

        Parameters
        ----------
        data : array-like
            The data to run the algorihthm on, can have almost any shape.

        Returns
        -------
        graph : The graph output from km.KeplerMapper(...).map(...)

        """
        # Initialize
        logging.info("Applying the mapping algorithm.")
        mapper = km.KeplerMapper(verbose=2)

        # We create a custom 1-D lens with Isolation Forest
        model = ensemble.IsolationForest()
        model.fit(data)
        isolation_forest = model.decision_function(data).reshape((data.shape[0], 1))

        # Fit to and transform the data
        tsne_projection = mapper.fit_transform(
            data,
            projection=sklearn.manifold.TSNE(
                n_components=2,
                perplexity=20,
                init='pca'
            )
        )

        lens = np.c_[isolation_forest, tsne_projection]

        # Create dictionary called 'graph' with nodes, edges and meta-information
        graph = mapper.map(tsne_projection,
                           coverer=km.Cover(10, 0.2),
                           clusterer=sklearn.cluster.DBSCAN(eps=1.0, min_samples=2))

        color_function = np.array([self._label_to_color(self.labels[i])
                                   for i in range(len(data))])
        # Visualize it
        mapper.visualize(graph,
                         path_html="actions.html",
                         title="chunk",
                         custom_tooltips=self.tooltips,
                         color_function=color_function)

        return graph

    def _label_to_color(self, label):
        max_value = 1000
        if label == 'scan':
            return 0
        elif label == 'cash':
            return max_value / 4
        elif label == 'moving':
            return (max_value / 4) * 2
        elif label == 'still':
            return (max_value / 4) * 3
        else:
            return max_value

    def create_tooltips(self, videos):
        """Creates tooltip videos for the chunks, writes them to output/tooltips.

        Parameters
        ----------
        videos : array-like, shape = [n_chunks, 1]
            The path to the videos corresponding to the chunks.

        """
        logging.info("Creating tooltip videos")
        self.tooltips = np.array([self._to_tooltip(videos[i], chunk, i, self.chunk_frames[i])
                                  for i, chunk in enumerate(self.chunks)])

    def _to_tooltip(self, video, chunk, chunk_index, frames):
        out_file_pose = os.path.join(
            'output/tooltips', "pose-{}".format(chunk_index) + '.avi')
        out_file_scene = os.path.join(
            'output/tooltips', "scene-{}".format(chunk_index) + '.avi')
        # self.chunk_visualiser.chunk_to_video_pose(chunk, out_file_pose, frames)
        # self.chunk_visualiser.chunk_to_video_scene(
        #     video, chunk, out_file_scene, frames, self.labels[chunk_index])

        tooltip = """
            <video
                controls
                loop
                width="90"
                height="90"
                autoplay
                src={}>
            </video>
        """.format(out_file_scene)

        return tooltip
