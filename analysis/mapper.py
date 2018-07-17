import cv2
import kmapper as km
import sklearn
from sklearn import ensemble
import numpy as np
from time import sleep
import sys
import os
import logging

from tracker import TrackVisualiser, Track, Person
from util import COCOKeypoints


class Mapper:

    def __init__(self, chunks, chunk_frames, original_chunks, frames_per_chunk):
        self.chunks = chunks
        self.chunk_frames = chunk_frames
        self.original_chunks = original_chunks
        self.frames_per_chunk = frames_per_chunk

    def visualise(self, video):
        capture = cv2.VideoCapture(video)

        nodes = self.graph['nodes']
        for name, node in nodes.items():
            print(name)
            self._draw_node(capture, node)
            sleep(1)

    def _draw_node(self, capture, node):
        visualiser = TrackVisualiser()

        for point in node:

            chunk_index = self.labels_int[point][0]
            person_index = self.labels_int[point][1]

            start_frame = self.labels_int[point][2]
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            original_chunk = self.original_chunks[person_index][chunk_index]
            translated_chunk = self.chunks[person_index][chunk_index]

            self._draw_chunk(original_chunk, translated_chunk, start_frame, visualiser)

    def mapper(self):
        arm_keypoint_indicies = [COCOKeypoints.RWrist.value, COCOKeypoints.LWrist.value,
                                 COCOKeypoints.RElbow.value, COCOKeypoints.LElbow.value]
        data = np.array([[p for person in action for k in person[arm_keypoint_indicies] for p in k[:2]]
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
        model = ensemble.IsolationForest()
        model.fit(data)
        lens1 = model.decision_function(data).reshape((data.shape[0], 1))

        # We create another 1-D lens with L2-norm
        mapper = km.KeplerMapper(verbose=3)
        lens2 = mapper.fit_transform(data, projection="l2norm")

        # Combine both lenses to create a 2-D [Isolation Forest, L^2-Norm] lens
        lens = np.c_[lens1, lens2]

        # Create the simplicial complex
        graph = mapper.map(lens,
                           data,
                           coverer=km.Cover(5, 0.3),
                           clusterer=sklearn.cluster.KMeans(n_clusters=2))
        # # Initialize
        # mapper = km.KeplerMapper(verbose=2)

        # # Fit to and transform the data
        # projected_data = mapper.fit_transform(data,
        #                                       projection=sklearn.manifold.TSNE())  # X-Y axis

        # # Create dictionary called 'graph' with nodes, edges and meta-information
        # graph = mapper.map(projected_data, coverer=km.Cover(10, 0.5))

        # # Visualize it
        mapper.visualize(graph, path_html="actions.html",
                         title="chunk",
                         custom_tooltips=tooltips)

        self.graph = graph
        self.labels_int = labels_int

        return graph, data, labels_int

    def _to_tooltip(self, chunk, person_index, chunk_index, start_frame):
        out_file = os.path.join(
            'output/tooltips', "{}-{}".format(person_index, chunk_index) + '.avi')
        self._chunk_to_video(chunk, out_file, start_frame)

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

    def _chunk_to_track(self, chunk, start_frame):
        track = Track()
        for i, p in enumerate(chunk):
            track.add_person(Person(p), i + start_frame)

        return track

    def _chunk_to_video(self, chunk, out_file, start_frame):
        translated_track = self._chunk_to_track(chunk, start_frame)

        frame_width = 100
        frame_height = 100
        fps = 10

        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        writer = cv2.VideoWriter(out_file, fourcc, fps, (frame_width, frame_height))
        visualiser = TrackVisualiser()
        for i in range(self.frames_per_chunk):
            translated_image = self._draw_translated_track(
                translated_track, i, start_frame, visualiser)
            translated_image = cv2.resize(translated_image, (100, 100))
            writer.write(translated_image)

        writer.release()

    def _draw_chunk(self, chunk, translated_chunk, start_frame, visualiser):
        track = self._chunk_to_track(original_chunk, start_frame)

        translated_track = self._chunk_to_track(translated_chunk, start_frame)

        for i in range(self.frames_per_chunk):
            success, original_image = capture.read()
            visualiser.draw_tracks([track], original_image, i + start_frame)
            visualiser.draw_frame_number(original_image, i + start_frame)

            translated_image = self._draw_translated_track(
                translated_track, i, start_frame, visualiser)

            cv2.imshow("output", original_image)
            cv2.imshow("translated_person", translated_image)
            cv2.waitKey(1)

    def _draw_translated_track(self, translated_track, i, start_frame, visualiser):
        blank_image = np.zeros((500, 500, 3), np.uint8)
        visualiser.draw_frame_number(blank_image, i + start_frame)
        visualiser.draw_people([translated_track], blank_image, i + start_frame)

        return blank_image
