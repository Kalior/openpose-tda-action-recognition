import numpy as np
import scipy.optimize
from time import time
import cv2

# Install openpose globally where all other python packages are installed.
from openpose import openpose as op

from .path_visualiser import PathVisualiser
from .person import Person
from .path import Path


class Tracker(object):

    def __init__(self, model_path='/models/', no_openpose=False):
        self.people_paths = []
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

        success, original_image = capture.read()
        while success:
            path_endpoints = [path.get_last_person() for path in self.people_paths]

            openpose_start_time = time()
            keypoints, image_with_keypoints = self.openpose.forward(original_image, True)
            people = self._convert_to_persons(keypoints)
            openpose_time = time() - openpose_start_time

            min_person_start_time = time()
            # Find out which people are closest to each other
            assignments, distances = self._find_assignments(people, path_endpoints)
            closest_person_time = time() - min_person_start_time

            self._update_paths(distances, assignments, people, path_endpoints)

            # Write the frame to a video
            self.visualiser.draw_paths(self.people_paths, image_with_keypoints)
            writer.write(image_with_keypoints)

            print("OpenPose: {:.5f}, Closest person: {:.5f}".format(
                openpose_time, closest_person_time))

            success, original_image = capture.read()

        capture.release()

    def _find_assignments(self, people, prev_people):
        # Pre-allocate the distance matrix
        distances = np.ndarray((len(prev_people), len(people)))

        # And calculate the distances...
        for i, prev_frame_person in enumerate(prev_people):
            for j, person in enumerate(people):
                distance = person.distance(prev_frame_person)
                distances[i, j] = distance

        # Find the best assignments between people in the two frames
        assignments = scipy.optimize.linear_sum_assignment(distances)
        return assignments, distances

    def _update_paths(self, distances, assignments, people, prev_people):
        # Special case for no assignments (either this frame has no people or no
        # previous frame had people)
        if assignments[0].size == 0 and assignments[1].size == 0:
            indicies = [self.person_counter + i for i in range(len(people))]
            assignments = [indicies, indicies]

        for from_, to in zip(assignments[0], assignments[1]):
            if from_ < len(prev_people):
                path_index = prev_people[from_].path_index
                avg_speed = self.people_paths[path_index].get_average_speed_in_window()

            # Make sure we know to which path the requested index belongs to
            #  and make sure there isn't a large gap between the two.
            if from_ >= len(prev_people) or distances[from_, to] >= avg_speed + 10:
                path_index = self.person_counter
                self.person_counter += 1

            # Extend people_paths if it's too short
            if len(self.people_paths) <= path_index:
                for _ in range(path_index - len(self.people_paths) + 1):
                    self.people_paths.append(Path([]))

            people[to].path_index = path_index
            self.people_paths[path_index].add_person(people[to])

    def _convert_to_persons(self, keypoints, keep_order=False):
        if keep_order:
            return [Person(k, i) for i, k in enumerate(keypoints)]
        else:
            return [Person(k) for k in keypoints]
