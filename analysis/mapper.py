import kmapper as km
import sklearn
from sklearn import ensemble
import numpy as np
import os
import logging

from util import COCOKeypoints
from .mapper_visualiser import MapperVisualiser


class Mapper:

    def __init__(self, chunks, chunk_frames, original_chunks, frames_per_chunk):
        self.chunks = chunks
        self.chunk_frames = chunk_frames
        self.original_chunks = original_chunks
        self.frames_per_chunk = frames_per_chunk
        self.selected_keypoints = [COCOKeypoints.RWrist.value, COCOKeypoints.LWrist.value,
                                   COCOKeypoints.RElbow.value, COCOKeypoints.LElbow.value]
        self.mapper_visualiser = MapperVisualiser(
            chunks, chunk_frames, original_chunks, frames_per_chunk, self.selected_keypoints)

    def visualise(self, video, graph, labels):
        self.mapper_visualiser.visualise(video, graph, labels)

    def mapper(self):
        data = np.array([[p for person in action for k in person[self.selected_keypoints] for p in k[:2]]
                         for path in self.chunks for action in path])

        labels = np.array(["Frame: {}, person: {}".format(self.chunk_frames[j][i], j)
                           for j, path in enumerate(self.chunks) for i, _ in enumerate(path)])
        labels_int = np.array([(i, j, self.chunk_frames[j][i])
                               for j, path in enumerate(self.chunks) for i, _ in enumerate(path)],
                              dtype=np.int)

        logging.info("Creating tooltip videos")
        tooltips = np.array([self._to_tooltip(action, j, i, self.chunk_frames[j][i])
                             for j, path in enumerate(self.chunks) for i, action in enumerate(path)])

        # We create a custom 1-D lens with Isolation Forest
        # model = ensemble.IsolationForest()
        # model.fit(data)
        # lens1 = model.decision_function(data).reshape((data.shape[0], 1))

        # # We create another 1-D lens with L2-norm
        # mapper = km.KeplerMapper(verbose=3)
        # lens2 = mapper.fit_transform(data, projection="l2norm")

        # # Combine both lenses to create a 2-D [Isolation Forest, L^2-Norm] lens
        # lens = np.c_[lens1, lens2]

        # # Create the simplicial complex
        # graph = mapper.map(lens1,
        #                    data,
        #                    coverer=km.Cover(10, 0.3),
        #                    clusterer=sklearn.cluster.KMeans(n_clusters=2))
        # # Initialize
        mapper = km.KeplerMapper(verbose=2)

        # Fit to and transform the data
        projected_data = mapper.fit_transform(data,
                                              projection=sklearn.manifold.TSNE())

        # Create dictionary called 'graph' with nodes, edges and meta-information
        graph = mapper.map(projected_data, coverer=km.Cover(5, 0.33),
                           clusterer=sklearn.cluster.KMeans(n_clusters=3))

        # # Visualize it
        mapper.visualize(graph, path_html="actions.html",
                         title="chunk",
                         custom_tooltips=tooltips)

        return graph, data, labels_int

    def _to_tooltip(self, chunk, person_index, chunk_index, start_frame):
        out_file = os.path.join(
            'output/tooltips', "{}-{}".format(person_index, chunk_index) + '.avi')
        self.mapper_visualiser.chunk_to_video(chunk, out_file, start_frame)

        tooltip = """
            <video
                controls
                loop
                width="90"
                height="90"
                autoplay
                src={}>
            </video>
        """.format(out_file)

        return tooltip
