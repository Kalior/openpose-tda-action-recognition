import cv2
import kmapper as km
import sklearn
import numpy as np
from time import sleep
import sys


from tracker import TrackVisualiser
from util import COCOKeypoints


class Mapper:

    def __init__(self, chunks, chunk_frames):
        self.chunks = chunks
        self.chunk_frames = chunk_frames

    def visualise(self, video):
        capture = cv2.VideoCapture(video)

        nodes = self.graph['nodes']
        for node in nodes.values():
            draw_node(capture, node)

    def draw_node(self, capture, node):
        visualiser = TrackVisualiser()

        print(node)
        for point in node:
            # coordinates = data[point]

            chunk_index = self.labels_int[point][0]
            person_index = self.labels_int[point][1]
            try:
                paths = self.chunks[person_index][chunk_index]
            except:
                print("size: {}, person: {}, chunk: {}".format(
                    self.chunks.shape, person_index, chunk_index))
                sys.exit(-1)

            start_frame = int(self.chunk_frames[person_index][chunk_index])

            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # paths = [np.stack([coordinates[j::8], coordinates[j + 1::8]], axis=1)
            #          for j in range(0, 8, 2)]
            track = Track()
            for i, p in enumerate(paths):
                track.add_person(Person(p), i + start_frame)

            for i in range(10):
                success, original_image = capture.read()
                visualiser.draw_tracks([track], original_image, i + start_frame)
                # for path in paths:
                #     path = [keypoints[0] for keypoints in path]
                #     print(path)
                #     visualiser._draw_path(original_image, path[:i], (255, 255, 255))
                visualiser.draw_frame_number(original_image, i + start_frame)

                cv2.imshow("output", original_image)
                cv2.waitKey(15)
                sleep(0.1)

    def mapper(self):
        arm_keypoint_indicies = [COCOKeypoints.RWrist.value, COCOKeypoints.LWrist.value,
                                 COCOKeypoints.RElbow.value, COCOKeypoints.LElbow.value]
        data = np.array([[p for person in action for k in person[arm_keypoint_indicies] for p in k[:2]]
                         for path in self.chunks for action in path])

        labels = np.array(["{}:{}".format(i, j)
                           for j, path in enumerate(self.chunks) for i, action in enumerate(path)])
        labels_int = np.array([(i, j)
                               for j, path in enumerate(self.chunks) for i, action in enumerate(path)])

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
