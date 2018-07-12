import numpy as np
import scipy.optimize
from time import time
import cv2
import os
import logging

# Install openpose globally where all other python packages are installed.
from openpose import openpose as op

# Tensorflow implementation of openpose:
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

from .path_visualiser import PathVisualiser
from .person import Person
from .path import Path

import util


class Tracker(object):

    def __init__(self, with_tf_openpose=False, model_path='/models/', no_openpose=False, only_track_arms=False, out_dir='output'):
        self.only_track_arms = only_track_arms
        Person.only_track_arms = only_track_arms

        self.people_paths = []
        self.with_tf_openpose = with_tf_openpose
        if with_tf_openpose:
            self.tf_openpose = TfPoseEstimator(get_graph_path("mobilenet_thin"),
                                               target_size=(432, 368))
        elif not no_openpose:
            # Initialise openpose
            params = self._openpose_parameters(model_path)
            # Construct OpenPose object allocates GPU memory
            self.openpose = op.OpenPose(params)
        else:
            # Used only for testing purposes.
            self.openpose = None

        self.speed_change_threshold = 10

        self.visualiser = PathVisualiser()

        self.out_dir = out_dir
        try:
            os.stat(out_dir)
        except:
            os.makedirs(out_dir)

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
        if self.with_tf_openpose:
            image_height, image_width = original_image.shape[:2]

            humans = self.tf_openpose.inference(original_image, resize_to_default=True)
            image_with_keypoints = TfPoseEstimator.draw_humans(
                original_image, humans, imgcopy=True)
            people = np.array([util.tf_openpose_human_to_np(human, image_width, image_height)
                               for human in humans])
            return people, image_with_keypoints
        else:
            keypoints, image_with_keypoints = self.openpose.forward(original_image, True)
            return keypoints, image_with_keypoints

    def video(self, file):
        capture = cv2.VideoCapture(file)
        self.speed_change_threshold = 10  # int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) / 10

        writer = self._create_writer(file, capture)

        current_frame = 0
        success, original_image = capture.read()
        while success:
            path_endpoints = [path.get_last_person()
                              for path in self.people_paths
                              if path.is_relevant(current_frame) and
                              path.get_last_person().is_relevant()]

            openpose_start_time = time()
            keypoints, image_with_keypoints = self._forward(original_image)
            people = [p for p in self._convert_to_persons(keypoints) if p.is_relevant()]
            openpose_time = time() - openpose_start_time

            min_person_start_time = time()
            # Find out which people are closest to each other
            assignments, distances, removed_people = self._find_assignments(
                people, path_endpoints, current_frame)

            #  Add back the people we couldn't associate well during the assignment process
            # to the back of the list
            people = people + removed_people

            self._update_paths(distances, assignments, people, path_endpoints, current_frame)
            closest_person_time = time() - min_person_start_time

            visualisation_start_time = time()
            self.visualiser.draw_paths(
                self.people_paths, image_with_keypoints, current_frame, self.only_track_arms)
            visualisation_time = time() - visualisation_start_time

            # Write the frame to a video
            writer.write(image_with_keypoints)

            logging.info("OpenPose: {:.5f}, Closest person: {:.5f}, Draw paths to img: {:.5f}".format(
                openpose_time, closest_person_time, visualisation_time))

            success, original_image = capture.read()
            current_frame += 1

        capture.release()
        writer.release()

    def _create_writer(self, in_file, capture):
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_code = capture.get(cv2.CAP_PROP_FOURCC)
        fps = int(capture.get(cv2.CAP_PROP_FPS))

        basename = os.path.basename(in_file)
        filename, _ = os.path.splitext(basename)
        out_file = os.path.join(self.out_dir, filename + '.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        writer = cv2.VideoWriter(out_file, fourcc, fps, (frame_width, frame_height))

        return writer

    def _find_assignments(self, people, prev_people, current_frame):
        # Pre-allocate the distance matrix
        distances = np.empty((len(prev_people), len(people)))

        # And calculate the distances...
        for i, prev_frame_person in enumerate(prev_people):
            for j, person in enumerate(people):
                distance = person.distance(prev_frame_person)
                distances[i, j] = distance

        removed_people = []
        # Find the best assignments between people in the two frames
        valid_assignment = False
        while not valid_assignment:
            assignments = scipy.optimize.linear_sum_assignment(distances)
            valid_assignment, distances, removed_person = self._is_assignment_valid(
                assignments, distances, people, prev_people, current_frame)
            if removed_person is not None:
                removed_people.append(removed_person)

        return assignments, distances, removed_people

    def _is_assignment_valid(self, assignments, distances, people, prev_people, current_frame):
        for from_, to in zip(assignments[0], assignments[1]):
            path_index = prev_people[from_].path_index
            avg_speed = self.people_paths[path_index].get_average_speed_in_window(10)
            frames_since_last_update = current_frame - \
                self.people_paths[path_index].last_frame_update

            #  If the movement is too large, assume that the new item can't
            # be associated well. (Which will force it to get a new path later
            # in the processing).
            if distances[from_, to] > avg_speed * frames_since_last_update + self.speed_change_threshold:
                logging.debug("Invalid association! from: {}, to: {}, dist: {}, avg_speed: {}, frames since last update: {}".format(
                    from_, to, distances[from_, to], avg_speed, frames_since_last_update))
                distances = np.delete(distances, to, axis=1)
                removed_person = people.pop(to)
                return False, distances, removed_person

        return True, distances, None

    def _update_paths(self, distances, assignments, people, prev_people, current_frame):
        for from_, to in zip(assignments[0], assignments[1]):
            logging.debug("From: {}, to: {}  people: {}  prev_people: {}".format(
                from_, to, len(people), len(prev_people)))
            path_index = self._establish_index_of_path(from_, to, prev_people, distances)

            people[to].path_index = path_index
            self.people_paths[path_index].add_person(people[to], current_frame)

        # If a person is not assigned to a path yet, assign it to a new path
        self._add_unassigned_people(assignments, people, current_frame)

    def _add_unassigned_people(self, assignments, people, current_frame):
        for i, _ in enumerate(people):
            if i not in assignments[1]:
                path = Path()
                people[i].path_index = len(self.people_paths)
                path.add_person(people[i], current_frame)
                self.people_paths.append(path)

    def _establish_index_of_path(self, from_, to, prev_people, distances):
        # Make sure we know to which path the requested index belongs to
        if from_ < len(prev_people):
            path_index = prev_people[from_].path_index
        else:
            path_index = len(self.people_paths)
            self.people_paths.append(Path())

        return path_index

    def _convert_to_persons(self, keypoints):
        return [Person(k) for k in keypoints]
