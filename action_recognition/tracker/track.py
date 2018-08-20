import numpy as np
import copy
import logging


class Track:
    """The track of a person through a video.

    Essentially a list of Person, with the corresponding frame number that the
    person was identified at. Also contains a lot of functions that operate on
    the Track for post processing.

    """

    def __init__(self):
        self.track = []
        self.frame_assigned = []
        self.last_frame_update = -1

    def __len__(self):
        return len(self.track)

    def __getitem__(self, item):
        return self.track[item]

    def copy(self, number_of_frames):
        new_track = Track()
        new_track.track = self.track[number_of_frames:]
        new_track.frame_assigned = self.frame_assigned[number_of_frames:]
        new_track.last_frame_update = new_track.frame_assigned[-1]

        return new_track

    def add_person(self, person, current_frame):
        """Adds a Person to the track
        Parameters
        ----------
        person : Person object
            The person to add to the track.
        current_frame : int
            The frame number the person was identified at.

        """
        self.last_frame_update = current_frame
        self.frame_assigned.append(current_frame)
        self.track.append(person)

    def get_last_person(self):
        """Retrieves the last person in the track
        Returns
        -------
        person : Person object

        """
        return self.track[-1]

    def get_average_speed_in_window(self, window_size=-1):
        """Gives average distance moved over a window of size window_size.

        Parameters
        ----------
        window_size : int, optional
            The number of frames to calculate the average. If unspecified,
            will calculate over the entire track.

        Returns
        -------
        speed : float
            The average speed calculated.
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
        """Paths are relevant if they are updated with the last 10 frames.

        Parameters
        ----------
        current_frame : int
            The current frame to compare the latest update of the track to.

        Returns
        -------
        relevant : boolean
        """
        return self.last_frame_update > current_frame - 10

    def get_keypoint_path(self, idx, current_frame=-1):
        """Returns the path of only the specified keypoint by idx.

        Parameters
        ----------
        idx : int, the keypoint index to get the path for.
        current_frame : int
            The current frame to get the path up until.

        Returns
        -------
        path : array-like
            shape = [n_frames_until_current, 2]

        """
        if current_frame == -1:
            current_frame = self.last_frame_update

        return [p[idx][:2] for f, p in zip(self.frame_assigned, self.track)
                if np.any(p[idx][:2]) and
                f <= current_frame]

    def get_keypoints_at(self, frame):
        """Returns all keypoints at time frame.

        Parameters
        ----------
        frame : int
            The current frame to get the keypoints of.

        Returns
        -------
        keypoints : array-like
            shape = [n_keypoints, 2]

        """

        path = [k[:, 2] for k, f in zip(self.track, self.frame_assigned)
                if f <= frame and
                np.any(k[:, :2])]

        if len(path) > 0:
            return path[-1]
        else:
            return None

    def divide_into_chunks(self, frames_per_chunk, overlap=-1):
        """Divides the track into chunks with overlaps, used for action recognition.

        Parameters
        ----------
        frames_per_chunks : int
            The number of frames each chunk should contain.
        overlap : int, optional
            The number of frames the chunks should overlap with.
            If not specified, half of frames_per_chunk is used.

        Returns
        -------
        chunks : array-like
            shape = [n_chunks, frames_per_chunk, n_keypoints, 3]
        frames : array-like
            shape = [n_chunks, frames_per_chunk, 1]
            The Frame number of each part of each chunk

        """
        if overlap == -1:
            overlap = int(frames_per_chunk / 2)

        number_of_chunks = int((len(self.track) - frames_per_chunk - 1) /
                               (frames_per_chunk - overlap) + 1)

        if number_of_chunks <= 0:
            return np.array([]), np.array([])

        keypoint_shape = self.track[0].keypoints.shape
        chunks = np.zeros((number_of_chunks, frames_per_chunk, *keypoint_shape))
        frames = np.zeros((number_of_chunks, frames_per_chunk), dtype=np.int)
        start_index = 0
        index = 0
        while start_index + frames_per_chunk < len(self.track):
            chunk, chunk_frames = self._chunk_from_index(start_index, frames_per_chunk)

            chunks[index] = chunk
            frames[index] = chunk_frames
            start_index += frames_per_chunk - overlap
            index += 1

        return chunks, frames

    def chunk_from_frame(self, start_frame, frames_per_chunk):
        """Gets a chunk from the start_frame with length of frames_per_chunk.

        Parameters
        ----------
        start_frame : int
            The start frame of the chunk
        frames_per_chunk : int
            The number of frames to include in the chunk

        Returns
        -------
        chunk : array-like
            Chunk of track. Shape = [frames_per_chunk, n_keypoints, 3]

        """
        start_index = 0
        # Iterate to where the chunk should start
        while start_index < len(self.frame_assigned) and self.frame_assigned[start_index] < start_frame:
            start_index += 1

        return self._chunk_from_index(start_index, frames_per_chunk)

    def _chunk_from_index(self, start_index, frames_per_chunk):
        end_index = start_index + frames_per_chunk
        chunk = np.array([p.keypoints for p in self.track[start_index:end_index]])
        frames = np.array(self.frame_assigned[start_index:end_index])
        return chunk, frames

    def to_np(self):
        """Converts the track into pure numpy arrays.

        Returns
        -------
        np_path : array-like
            Shape = [len(track), n_keypoints, 3]
        np_frames : array-like
            Shape = [len(track), 1]
        """
        np_path = np.array([p.keypoints for p in self.track])
        np_frames = np.array(self.frame_assigned)

        return np_path, np_frames

    def combine(self, other):
        """Joins this track with other.

        Deals with cases where the two tracks have people at the same frame,
        and when one track follows the other.

        Parameters
        ----------
        other : Track
            The other track object to combine with this one.

        """
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
        """Checks if other overlaps with this track.

        More specifically, it check if the two tracks are nearby each other
        for all of their frames.

        Parameters
        ----------
        other : Track
            The other track object to compare to this one.

        Returns
        -------
        overlaps : boolean
            True if the two tracks are nearby each other.

        """
        if self.frame_assigned[-1] < other.frame_assigned[0] or \
                self.frame_assigned[0] > other.frame_assigned[-1]:
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

    def fill_missing_keypoints(self, fill_type='copy'):
        """Fill missing keypoints with data from previous parts of the track

        Parameters
        ----------
        fill_type : str, optional, default = 'copy'
            Either 'copy' or 'diff', as explained in Person.fill_missing_keypoints()

        """
        if len(self.track) < 2:
            return

        for i in range(1, len(self.track)):
            self.track[i].fill_missing_keypoints(self.track[i - 1], fill_type)

    def remove_frame_duplicates(self):
        """Removes parts of the track where two parts were assigned to the same frame number.

        """
        i = 0
        while i < len(self.frame_assigned) - 1:
            if self.frame_assigned[i] == self.frame_assigned[i + 1]:
                self.frame_assigned.pop(i + 1)
                self.track.pop(i + 1)
            else:
                i += 1

    def fill_missing_frames(self):
        """Fills frames that aren't assigned a value with interpolated points.

        Specifically, it identifies gaps in the frame numbers where there are
        no assignments, and interpolates values for the keypoints from the
        next identified frame and the previous identified frame.

        """
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

    def reset_keypoints(self):
        """Resets the Keypoints in every Person in track to the original.
        """
        for person in self.track:
            person.reset_keypoints()
