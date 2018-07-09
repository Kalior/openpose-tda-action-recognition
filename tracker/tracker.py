import numpy as np
import scipy.optimize
from time import time
import cv2

# Install openpose globally where all other python packages are installed.
from openpose import openpose as op

from .path_visualiser import PathVisualiser
from .person import Person


class Tracker(object):

    def __init__(self, model_path='/models/', no_openpose=False):
        self.people = []
        self.people_paths = []
        self.path_indices = {}
        if not no_openpose:
            # Initialise openpose
            params = self._openpose_parameters(model_path)
            # Construct OpenPose object allocates GPU memory
            self.openpose = op.OpenPose(params)
        else:
            self.openpose = None

        self.person_counter = 0

        self.visualiser = PathVisualiser()

    def _openpose_parameters(self, model_path):
        params = {
            "logging_level": 3,
            "output_resolution": "-1x-1",
            "net_resolution": "-1x192",
            "model_pose": "COCO",
            "alpha_pose": 0.6,
            "scale_gap": 0.3,
            "scale_number": 1,
            "render_threshold": 0.05,
            # If GPU version is built, and multiple GPUs are available, set the ID here
            "num_gpu_start": 0,
            "disable_blending": False,
            # Ensure you point to the correct path where models are located
            "default_model_folder": model_path
        }
        return params

    def video(self, file):
        capture = cv2.VideoCapture(file)

        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_code = capture.get(cv2.CAP_PROP_FOURCC)
        fps = int(capture.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

        prev_frame_people = []

        success, original_image = capture.read()
        while success:
            openpose_start_time = time()
            keypoints, image_with_keypoints = self.openpose.forward(original_image, True)
            people = self._convert_to_persons(keypoints)
            openpose_time = time() - openpose_start_time

            min_person_start_time = time()
            # Find out which people are closest to each other
            assignments, distances = self._find_assignments(prev_frame_people, people)
            closest_person_time = time() - min_person_start_time

            self._update_paths(distances, assignments, people)

            self.visualiser.draw_paths(self.people_paths, image_with_keypoints)
            writer.write(image_with_keypoints)

            print("OpenPose: {:.5f}, Closest person: {:.5f}".format(
                openpose_time, closest_person_time))

            prev_frame_people = people
            success, original_image = capture.read()

        capture.release()

    def _find_assignments(self, prev_frame_people, people):
        # Pre-allocate the distance matrix
        distances = np.ndarray((len(prev_frame_people), len(people)))

        # And calculate the distances...
        for i, prev_frame_person in enumerate(prev_frame_people):
            for j, person in enumerate(people):
                distance = person.distance(prev_frame_person)
                distances[i, j] = distance

        # Find the best assignments between people in the two frames
        assignments = scipy.optimize.linear_sum_assignment(distances)
        return assignments, distances

    def _update_paths(self, distances, assignments, people):
        new_path_indices = {}
        for from_, to in zip(assignments[0], assignments[1]):
            # Make sure we know to which path the requested index belongs to
            #  and make sure there isn't a large gap between the two.
            if from_ in self.path_indices and distances[from_, to] < 10:
                path_index = self.path_indices[from_]
            else:
                path_index = self.person_counter
                self.person_counter += 1

            # Extend people_paths if it's too short
            if len(self.people_paths) <= path_index:
                for _ in range(path_index - len(self.people_paths) + 1):
                    self.people_paths.append([])

            people[to].path_index = path_index
            self.people_paths[path_index].append(people[to])

            new_path_indices[to] = path_index

        self.path_indices = new_path_indices

    def _convert_to_persons(self, keypoints, keep_order=False):
        if keep_order:
            return [Person(k, i) for i, k in enumerate(keypoints)]
        else:
            return [Person(k) for k in keypoints]
