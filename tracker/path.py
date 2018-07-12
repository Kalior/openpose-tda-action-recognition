import numpy as np


class Path(object):

    def __init__(self):
        self.path = []
        self.last_frame_update = -1

    def __len__(self):
        return len(self.path)

    def __getitem__(self, item):
        return self.path[item]

    def add_person(self, person, current_frame):
        self.last_frame_update = current_frame
        self.path.append(person)

    def get_last_person(self):
        return self.path[-1]

    def get_average_speed_in_window(self, window_size=-1):
        """
            Gives distance moved (in pixels I think) per frame
        """
        if window_size == -1:
            window_size = len(self.path)

        if len(self.path) < 2:
            return 10

        distance = 0
        start_index = max(1, len(self.path) - window_size + 1)
        for i in range(start_index, len(self.path)):
            distance += self.path[i - 1].distance(self.path[i])

        speed = distance / (len(self.path) - start_index)
        return speed

    def is_relevant(self, current_frame):
        # Paths are relevant if they are used recently
        return self.last_frame_update > current_frame - 10

    def get_keypoint_path(self, idx):
        return [p[idx][:2] for p in self.path if np.any(p[idx][:2])]

    def divide_into_chunks(self, frames_per_chunk, overlap=-1):
        if overlap == -1:
            overlap = int(frames_per_chunk / 2)

        #  Turn chunks into np array and preallocate if this turns into a speed
        # issue.
        chunks = []
        start_index = 0
        while start_index + frames_per_chunk < len(self.path):
            chunk = np.array([p.keypints for p in self.path[start_index:frames_per_chunk]])
            chunks.append(chunk)
            start_index = frames_per_chunk - overlap
        return chunks
