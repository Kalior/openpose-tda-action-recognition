import numpy as np
import scipy.optimize
from time import time
import cv2
import os

# Install openpose globally where all other python packages are installed.
from openpose import openpose as op

# Tensorflow implementation of openpose:
from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

from .path_visualiser import PathVisualiser
from .person import Person
from .path import Path


class Tracker(object):

    def __init__(self, tf_openpose=False, model_path='/models/', no_openpose=False):
        self.people_paths = []
        self.tf_openpose = tf_openpose
        if tf_openpose:
            self.e = TfPoseEstimator(get_graph_path("mobilenet_thin"), target_size=(432, 368))
        elif not no_openpose:
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

    def _forward(self, original_image):
        if self.tf_openpose:
            image_height, image_width = original_image.shape[:2]

            humans = self.e.inference(original_image)
            image_with_keypoints = TfPoseEstimator.draw_humans(
                original_image, humans, imgcopy=False)
            people = np.array([self._tf_openpose_human_to_np(human, image_width, image_height)
                               for human in humans])
            return people, image_with_keypoints
        else:
            keypoints, image_with_keypoints = self.openpose.forward(original_image, True)
            return keypoints, image_with_keypoints

    def _tf_openpose_human_to_np(self, human, image_width, image_height):
        keypoints = np.empty([common.CocoPart.Background.value, 3])
        for i in range(common.CocoPart.Background.value):
            if i not in human.body_parts.keys():
                keypoints[i] = np.array([0, 0, 0])
            else:
                part = human.body_parts[i]
                x = part.x * image_width + 0.5
                y = part.y * image_height + 0.5
                confidence = part.score
                keypoints[i] = np.array([x, y, confidence])
        return keypoints

    def video(self, file, only_arms=False):
        capture = cv2.VideoCapture(file)

        writer = self._create_writer(file, capture)

        current_frame = 0
        success, original_image = capture.read()
        while success:
            path_endpoints = [path.get_last_person()
                              for path in self.people_paths
                              if path.is_relevant(current_frame)]

            openpose_start_time = time()
            keypoints, image_with_keypoints = self._forward(original_image)
            people = self._convert_to_persons(keypoints)
            openpose_time = time() - openpose_start_time

            min_person_start_time = time()
            # Find out which people are closest to each other
            assignments, distances = self._find_assignments(people, path_endpoints, only_arms)
            closest_person_time = time() - min_person_start_time

            self._update_paths(distances, assignments, people, path_endpoints, current_frame)

            self.visualiser.draw_paths(
                self.people_paths, image_with_keypoints, current_frame, only_arms)
            # Write the frame to a video
            writer.write(image_with_keypoints)

            print("OpenPose: {:.5f}, Closest person: {:.5f}".format(
                openpose_time, closest_person_time))

            success, original_image = capture.read()
            current_frame += 1

        capture.release()
        writer.release()

    def _create_writer(self, in_file, capture):
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_code = capture.get(cv2.CAP_PROP_FOURCC)
        fps = int(capture.get(cv2.CAP_PROP_FPS))

        filename = os.path.basename(in_file)
        out_file = os.path.join("output", filename + '.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        writer = cv2.VideoWriter(out_file, fourcc, fps, (frame_width, frame_height))

        return writer

    def _find_assignments(self, people, prev_people, only_arms=False):
        # Pre-allocate the distance matrix
        distances = np.empty((len(prev_people), len(people)))

        # And calculate the distances...
        for i, prev_frame_person in enumerate(prev_people):
            for j, person in enumerate(people):
                distance = person.distance(prev_frame_person, only_arms)
                distances[i, j] = distance

        # Find the best assignments between people in the two frames
        assignments = scipy.optimize.linear_sum_assignment(distances)
        return assignments, distances

    def _update_paths(self, distances, assignments, people, prev_people, current_frame):
        # Special case for no assignments (either this frame has no people or no
        # previous frame had people)
        if assignments[0].size == 0 and assignments[1].size == 0:
            from_indicies = [self.person_counter + i for i in range(len(people))]
            to_indicies = [i for i in range(len(people))]
            assignments = [from_indicies, to_indicies]

        unassigned_people_to = [i for i, _ in enumerate(people) if i not in assignments[1]]
        unassigned_people_from = [i + self.person_counter for i in unassigned_people_to]

        from_assignments = np.append(assignments[0],
                                     np.array(unassigned_people_from, dtype=np.int))
        to_assignments = np.append(assignments[1], np.array(unassigned_people_to, dtype=np.int))

        for from_, to in zip(from_assignments, to_assignments):
            path_index = self._establish_index_of_path(from_, to, prev_people, distances)

            # Extend people_paths if it's too short
            self._extend_people_path_to(path_index)

            people[to].path_index = path_index
            self.people_paths[path_index].add_person(people[to], current_frame)

    def _establish_index_of_path(self, from_, to, prev_people, distances):
        if from_ < len(prev_people):
            path_index = prev_people[from_].path_index
            avg_speed = self.people_paths[path_index].get_average_speed_in_window()

        # Make sure we know to which path the requested index belongs to
        #  and make sure there isn't a large gap between the two.
        if from_ >= len(prev_people) or distances[from_, to] >= avg_speed + 10:
            path_index = self.person_counter
            self.person_counter += 1

        return path_index

    def _extend_people_path_to(self, new_length):
        if len(self.people_paths) <= new_length:
            for _ in range(new_length - len(self.people_paths) + 1):
                self.people_paths.append(Path())

    def _convert_to_persons(self, keypoints):
        return [Person(k) for k in keypoints]
