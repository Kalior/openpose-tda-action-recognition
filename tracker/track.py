import numpy as np


class Track(object):

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

        return [p[idx][:2] for i, p in enumerate(self.track) if np.any(p[idx][:2]) and self.frame_assigned[i] <= current_frame]

    def divide_into_chunks(self, frames_per_chunk, overlap=-1):
        if overlap == -1:
            overlap = int(frames_per_chunk / 2)

        #  Turn chunks into np array and preallocate if this turns into a speed
        # issue.
        chunks = []
        start_index = 0
        while start_index + frames_per_chunk < len(self.track):
            chunk = np.array([p.keypoints for p in self.track[start_index:frames_per_chunk]])
            chunks.append(chunk)
            start_index = frames_per_chunk - overlap
        return np.array(chunks)

    def to_np(self):
        np_path = np.array([p.keypoints for p in self.track])
        np_frames = np.array(self.frame_assigned)

        return np_path, np_frames
