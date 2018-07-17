import numpy as np

import logging


class Track:

    def __init__(self):
        self.track = []
        self.frame_assigned = []
        self.last_frame_update = -1

    def __len__(self):
        return len(self.track)

    def __getitem__(self, item):
        return self.track[item]

    def add_person(self, person, current_frame):
        self.last_frame_update = current_frame
        self.frame_assigned.append(current_frame)
        self.track.append(person)

    def get_last_person(self):
        return self.track[-1]

    def get_average_speed_in_window(self, window_size=-1):
        """
            Gives distance moved (in pixels I think) per frame
        """
        if window_size == -1:
            window_size = len(self.track)

        if len(self.track) < 2:
            return 10

        distance = 0
        start_index = max(1, len(self.track) - window_size + 1)
        for i in range(start_index, len(self.track)):
            distance += self.track[i - 1].distance(self.track[i])

        speed = distance / (len(self.track) - start_index)
        return speed

    def is_relevant(self, current_frame):
        # Paths are relevant if they are used recently
        return self.last_frame_update > current_frame - 10

    def get_keypoint_path(self, idx, current_frame=-1):
        if current_frame == -1:
            current_frame = self.last_frame_update

        return [p[idx][:2] for i, p in enumerate(self.track)
                if np.any(p[idx][:2]) and self.frame_assigned[i] <= current_frame]

    def divide_into_chunks(self, frames_per_chunk, overlap=-1):
        if overlap == -1:
            overlap = int(frames_per_chunk / 2)

        number_of_chunks = int((len(self.track) - frames_per_chunk - 1) /
                               (frames_per_chunk - overlap) + 1)
        if number_of_chunks <= 0:
            return np.array([])

        number_of_keypoints = self.track[0].keypoints.shape[0]
        values_per_keypoint = self.track[0].keypoints.shape[1]
        chunks = np.zeros((number_of_chunks, frames_per_chunk,
                           number_of_keypoints, values_per_keypoint))
        chunk_start_frames = np.zeros(number_of_chunks)
        start_index = 0
        index = 0
        while start_index + frames_per_chunk < len(self.track):
            chunk = np.array([p.keypoints for p in self.track[
                             start_index:(start_index + frames_per_chunk)]])
            chunks[index] = chunk
            chunk_start_frames[index] = self.frame_assigned[start_index]
            start_index += frames_per_chunk - overlap
            index += 1

        return chunks, chunk_start_frames

    def to_np(self):
        np_path = np.array([p.keypoints for p in self.track])
        np_frames = np.array(self.frame_assigned)

        return np_path, np_frames

    # Standard merge algorithm
    def combine(self, other):
        new_track = []
        new_frame_assigned = []
        self_index = 0
        other_index = 0

        # Join while it's still relevant to consider both lists
        # Check the frame they were added for the ordering of the
        # new track
        while self_index < len(self.track) and other_index < len(other.track):
            if self.frame_assigned[self_index] < other.frame_assigned[other_index]:
                new_track.append(self.track[self_index])
                new_frame_assigned.append(self.frame_assigned[self_index])
                self_index += 1
            else:
                new_track.append(other.track[other_index])
                new_frame_assigned.append(other.frame_assigned[other_index])
                other_index += 1

        # Add the entire rest of the lists
        if self_index >= len(self.track):
            new_track.extend(other.track[other_index:])
            new_frame_assigned.extend(other.frame_assigned[other_index:])
        elif other_index >= len(other.track):
            new_track.extend(self.track[self_index:])
            new_frame_assigned.extend(self.frame_assigned[self_index:])

        self.last_frame_update = new_frame_assigned[-1]
        self.track = new_track
        self.frame_assigned = new_frame_assigned
        self.remove_frame_duplicates()

    def overlaps(self, other):
        # print("Checking overlap")
        if self.frame_assigned[-1] < other.frame_assigned[0] or \
                self.frame_assigned[0] > other.frame_assigned[-1]:
            # print("Can't overlap")
            return False

        start_frame = max(self.frame_assigned[0], other.frame_assigned[0])

        self_index = self._find_start_index(start_frame, self.frame_assigned)
        other_index = self._find_start_index(start_frame, other.frame_assigned)

        #  If any of the shifts between the two paths have a distance above a threshold
        # we don't consider the two paths to be overlapping.
        while self_index < len(self.track) and other_index < len(other.track):
            if self.frame_assigned[self_index] < other.frame_assigned[other_index]:
                if self._check_frame_distance(self.frame_assigned, self.track, self_index,
                                              other.frame_assigned, other.track, other_index):
                    self_index += 1
                else:
                    return False
            else:
                if self._check_frame_distance(other.frame_assigned, other.track, other_index,
                                              self.frame_assigned, self.track, self_index):
                    other_index += 1
                else:
                    return False

        return True

    def _check_frame_distance(self, frame_assigned, track, index, other_frame_assigned, other_track, other_index):
        distance_threshold = 16

        frame_diff = frame_assigned[index] - other_frame_assigned[other_index]
        if index + 1 < len(frame_assigned):
            next_frame_diff = frame_assigned[index + 1] - other_frame_assigned[other_index]
        else:
            next_frame_diff = frame_diff - 1

        distance = track[index].distance(other_track[other_index])
        # print("Frames: {}, {}, indicies: {}, {}, distance: {}".format(
        #     frame_assigned[index],
        #     other_frame_assigned[other_index],
        #     index,
        #     other_index,
        #     distance))

        if next_frame_diff > frame_diff and next_frame_diff <= 0:
            return True
        else:
            return distance < distance_threshold

    def _find_start_index(self, start_frame, frame_assigned):
        index = 0
        # Find the frame where the two paths start overlapping
        while frame_assigned[index] < start_frame:
            index += 1
        index -= 1

        return max(0, index)

    # Fill missing keypoints with data from previous parts of the track
    def fill_missing_keypoints(self):
        if len(self.track) < 2:
            return

        for i in range(1, len(self.track)):
            self.track[i].fill_missing_keypoints(self.track[i - 1])

    def remove_frame_duplicates(self):
        i = 0
        while i < len(self.frame_assigned) - 1:
            if self.frame_assigned[i] == self.frame_assigned[i + 1]:
                self.frame_assigned.pop(i + 1)
                self.track.pop(i + 1)
            else:
                i += 1

    def fill_missing_frames(self):
        new_track = []
        new_frame_assigned = []

        new_track.append(self.track[0])
        new_frame_assigned.append(self.frame_assigned[0])

        for i in range(1, len(self.track)):
            frame_diff = self.frame_assigned[i] - self.frame_assigned[i - 1]
            if frame_diff == 1:
                new_track.append(self.track[i])
                new_frame_assigned.append(self.frame_assigned[i])
            else:
                interpolated_people = self.track[i].interpolate(self.track[i - 1], frame_diff)
                for j, person in enumerate(interpolated_people):
                    new_track.append(person)
                    new_frame_assigned.append(self.frame_assigned[i - 1] + j + 1)

                new_track.append(self.track[i])
                new_frame_assigned.append(self.frame_assigned[i])

        self.track = new_track
        self.frame_assigned = new_frame_assigned

    def translate_to_origin(self):
        for person in self.track:
            person.translate_to_origin()
