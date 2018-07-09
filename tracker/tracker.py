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
        success, img = capture.read()

        prev_frame_people, _ = self.openpose.forward(img, True)
        success, img = capture.read()

        # Keep track of which indices belong to which people, since the lists
        # aren't sorted in any way.
        self.path_indices = {i: i for i, _ in enumerate(prev_frame_people)}
        self.person_paths = [[Person(i, p)] for i, p in enumerate(prev_frame_people)]
        self.person_counter = len(prev_frame_people)

        for _ in range(10):
            openpose_start_time = time()
            people, output_image = self.openpose.forward(img, True)
            openpose_time = time() - openpose_start_time

            min_person_start_time = time()
            # Find out which people are closest to each other
            assignments, distances = self._find_assignments(prev_frame_people, people)
            closest_person_time = time() - min_person_start_time

            self._update_paths(distances, assignments, people)

            self.visualiser.draw_paths(self.people_paths, output_image)

            print("OpenPose: {:.5f}, Closest person: {:.5f}".format(
                openpose_time, closest_person_time))

            prev_frame_people = people
            success, img = capture.read()

    def _find_assignments(self, prev_frame_people, people):
        # Pre-allocate the distance matrix
        distances = np.ndarray((prev_frame_people.shape[0], people.shape[0]))

        # And calculate the distances...
        for i, prev_frame_person in enumerate(prev_frame_people):
            for j, person in enumerate(people):
                distance = self._person_distance(person, prev_frame_person)
                distances[i, j] = distance

        # Find the best assignments between people in the two frames
        assignments = scipy.optimize.linear_sum_assignment(distances)
        return assignments, distances

    def _update_paths(self, distances, assignments, people):
        new_path_indices = {}
        for from_, to in zip(assignments[0], assignments[1]):
            # Make sure we know to which path the requested index belongs to
            if from_ in self.path_indices and distances[from_, to] < 10:
                path_index = self.path_indices[from_]
            else:
                path_index = self.person_counter
                self.person_counter += 1

            # Extend people_paths if it's too short
            if len(self.people_paths) <= path_index:
                for _ in range(path_index - len(self.people_paths) + 1):
                    self.people_paths.append([])

            new_person = Person(path_index, people[to])
            self.people_paths[path_index].append(new_person)

            new_path_indices[to] = path_index

        self.path_indices = new_path_indices

    # A person is a [#keypoints x 3] numpy array
    # With [X, Y, Confidence] as the values.
    def _person_distance(self, person, prev_frame_person):
        # Disregard the confidence for now.
        xy_person = person[:, :2]
        xy_prev = prev_frame_person[:, :2]

        #   Don't include the keypoints we didn't identify
        # as this can give large frame-to-frame errors.
        xy_person, xy_prev = self._filter_nonzero(xy_person, xy_prev)

        if xy_person.size == 0:
            return 10000  # np.inf, but np.inf doesn't play nice with scipy.optimize

        # Calculate average distance between the two people
        distance = np.linalg.norm(xy_person - xy_prev)
        distance = distance / xy_person.size

        return distance

    def _filter_nonzero(self, first, second):
        first, second = first[np.nonzero(first)], second[np.nonzero(first)]
        first, second = first[np.nonzero(second)], second[np.nonzero(second)]
        return first, second
