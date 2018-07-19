import kmapper as km
import sklearn
from sklearn import ensemble
import numpy as np
import os
import logging

from util import COCOKeypoints
from .mapper_visualiser import MapperVisualiser


class Mapper:

    def __init__(self, chunks, chunk_frames, frames_per_chunk, video, labels):
        self.chunks = chunks
        self.chunk_frames = chunk_frames
        self.frames_per_chunk = frames_per_chunk
        self.selected_keypoints = [
            COCOKeypoints.RWrist.value,
            COCOKeypoints.LWrist.value,
            COCOKeypoints.RElbow.value,
            COCOKeypoints.LElbow.value
        ]
        self.translated_chunks = self._translate_chunks_to_origin()
        self.mapper_visualiser = MapperVisualiser(
            chunks, chunk_frames, self.translated_chunks, frames_per_chunk, self.selected_keypoints, video)
        self.labels = labels

    def visualise(self, graph, labels):
        self.mapper_visualiser.visualise(graph, labels)

    def mapper(self):
        logging.info("Flattening data into a datapoint per chunk of a track.")
        data, labels = self._flatten_chunks()
        # data, labels = self._velocity_of_chunks()

        logging.info("Creating tooltip videos")
        tooltips = np.array([self._to_tooltip(action, j, i, self.chunk_frames[j][i])
                             for j, path in enumerate(self.chunks) for i, action in enumerate(path)])

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
                perplexity=50,
                init='pca'
            )
        )

        lens = np.c_[isolation_forest, tsne_projection]

        # Create dictionary called 'graph' with nodes, edges and meta-information
        graph = mapper.map(tsne_projection,
                           coverer=km.Cover(10, 0.2),
                           clusterer=sklearn.cluster.DBSCAN(eps=1.0, min_samples=10))

        color_function = np.array([self._label_to_color(self.labels[str(i)])
                                   for i in range(len(data))])
        # Visualize it
        mapper.visualize(graph,
                         path_html="actions.html",
                         title="chunk",
                         custom_tooltips=tooltips,
                         color_function=color_function)

        return graph, labels

    def _label_to_color(self, label):
        if label == 'scan':
            return 0
        elif label == 'cash':
            return 255 / 4
        elif label == 'moving':
            return (255 / 4) * 2
        elif label == 'still':
            return (255 / 4) * 3
        else:
            return 255

    def _to_tooltip(self, chunk, person_index, chunk_index, start_frame):
        out_file_pose = os.path.join(
            'output/tooltips', "pose-{}-{}".format(person_index, chunk_index) + '.avi')
        out_file_scene = os.path.join(
            'output/tooltips', "scene-{}-{}".format(person_index, chunk_index) + '.avi')
        # self.mapper_visualiser.chunk_to_video_pose(chunk, out_file_pose, start_frame)
        self.mapper_visualiser.chunk_to_video_scene(
            chunk, out_file_scene, start_frame, self.labels[str(chunk_index)])

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

    def _flatten_chunks(self):
        data = np.array([action[:, self.selected_keypoints, :2].flatten()
                         for path in self.chunks for action in path])

        labels = np.array([(i, j, self.chunk_frames[j][i])
                           for j, path in enumerate(self.chunks) for i, _ in enumerate(path)],
                          dtype=np.int)

        return data, labels

    def _velocity_of_chunks(self):
        data = np.array([self._relative_velocity_of_chunk(action)
                         for path in self.chunks for action in path])

        labels = np.array([(i, j, self.chunk_frames[j][i])
                           for j, path in enumerate(self.chunks) for i, _ in enumerate(path)],
                          dtype=np.int)

        return data, labels

    def _relative_velocity_of_chunk(self, chunk):
        velocity = np.empty(
            (chunk.shape[0] - 1, len(self.selected_keypoints), 2))

        for i in range(1, len(chunk)):
            for j, keypoint_index in enumerate(self.selected_keypoints):
                keypoint = chunk[i, keypoint_index]
                prev_keypoint = chunk[i - 1, keypoint_index]
                velocity[i - 1, j] = prev_keypoint[:2] - keypoint[:2]

        return velocity.flatten()

    def _translate_chunks_to_origin(self):
        translated_chunks = np.zeros(self.chunks.shape, dtype=object)

        for i, track in enumerate(self.chunks):
            track = np.copy(track)
            for j, chunk in enumerate(track):
                self._normalise_chunk(chunk)
            translated_chunks[i] = track

        return translated_chunks

    def _normalise_chunk(self, chunk):
        # Don't take unidentified keypoints into account:
        mean = chunk[~np.all(chunk == 0, axis=2)].mean(axis=0)

        chunk[~np.all(chunk == 0, axis=2)] -= mean
