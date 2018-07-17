import cv2
import kmapper as km
import sklearn
import numpy as np
from time import sleep
import sys


from tracker import TrackVisualiser, Track, Person
from util import COCOKeypoints


class Mapper:

    def __init__(self, chunks, chunk_frames, original_chunks):
        self.chunks = chunks
        self.chunk_frames = chunk_frames
        self.original_chunks = original_chunks

    def visualise(self, video):
        capture = cv2.VideoCapture(video)

        nodes = self.graph['nodes']
        for node in nodes.values():
            self._draw_node(capture, node)

    def _draw_node(self, capture, node):
        visualiser = TrackVisualiser()

        print(node)
        for point in node:

            chunk_index = self.labels_int[point][0]
            person_index = self.labels_int[point][1]
            start_frame = self.labels_int[point][2]

            paths = self.original_chunks[person_index][chunk_index]
            translated_paths = self.chunks[person_index][chunk_index]

            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            track = Track()
            for i, p in enumerate(paths):
                track.add_person(Person(p), i + start_frame)

            translated_track = Track()
            for i, p in enumerate(translated_paths):
                translated_track.add_person(Person(p), i + start_frame)

            for i in range(10):
                success, original_image = capture.read()
                visualiser.draw_tracks([track], original_image, i + start_frame)
                # for path in paths:
                #     path = [keypoints[0] for keypoints in path]
                #     print(path)
                #     visualiser._draw_path(original_image, path[:i], (255, 255, 255))
                visualiser.draw_frame_number(original_image, i + start_frame)

                blank_image = np.zeros((500, 500))
                visualiser.draw_frame_number(blank_image, i + start_frame)
                visualiser.draw_people([translated_track], blank_image, i + start_frame)

                cv2.imshow("output", original_image)
                cv2.imshow("translated_person", blank_image)
                cv2.waitKey(1)

    def mapper(self):
        arm_keypoint_indicies = [COCOKeypoints.RWrist.value, COCOKeypoints.LWrist.value,
                                 COCOKeypoints.RElbow.value, COCOKeypoints.LElbow.value]
        data = np.array([[p for person in action for k in person for p in k[:2]]
                         for path in self.chunks for action in path])

        labels = np.array(["Frame: {}, person: {}".format(self.chunk_frames[j][i], j)
                           for j, path in enumerate(self.chunks) for i, _ in enumerate(path)])
        labels_int = np.array([(i, j, self.chunk_frames[j][i])
                               for j, path in enumerate(self.chunks) for i, _ in enumerate(path)],
                              dtype=np.int)

        # Initialize
        mapper = km.KeplerMapper(verbose=2)

        # Fit to and transform the data
        projected_data = mapper.fit_transform(data, projection=sklearn.manifold.TSNE())  # X-Y axis

        # Create dictionary called 'graph' with nodes, edges and meta-information
        graph = mapper.map(projected_data, coverer=km.Cover(10, 0.5))

        print(graph.keys())

        # Visualize it
        mapper.visualize(graph, path_html="actions.html",
                         title="chunk",
                         custom_tooltips=labels)

        self.graph = graph
        self.labels_int = labels_int

        return graph, data, labels_int
