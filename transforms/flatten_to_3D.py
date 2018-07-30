import numpy as np


class FlattenTo3D:

    def __init__(self, selected_keypoints, connect_keypoints):
        self.selected_keypoints = selected_keypoints
        self.connect_keypoints = connect_keypoints

    def transform(self, chunks):
        data = np.array([self._chunk_to_3D(chunk) for chunk in chunks])

        return data

    def _chunk_to_3D(self, chunk):
        number_of_frames = chunk.shape[0]
        flat_chunk = np.vstack(self._connect_keypoints(
            chunk[:, self.selected_keypoints, :2], 2))
        third_dimension = np.repeat(np.arange(0, number_of_frames), len(
            self.selected_keypoints) + len(connect_keypoints) * 2)
        return np.c_[flat_chunk, third_dimension]

    def _connect_keypoints(self, chunk, number_of_points=3):
        # return chunk
        new_number_of_keypoints = chunk.shape[1] + len(self.connect_keypoints) * number_of_points
        connected_chunk = np.zeros((chunk.shape[0], new_number_of_keypoints, chunk.shape[2]))
        connected_chunk[:, :chunk.shape[1]] = chunk
        for i, frame in enumerate(chunk):
            for j, (from_, to) in enumerate(self.connect_keypoints):
                start_index = j * number_of_points + chunk.shape[1]
                intermediate_points = self._intermediate_points(frame, from_, to, number_of_points)
                for k, points in enumerate(intermediate_points):
                    connected_chunk[i, start_index + k] = points

        return connected_chunk

    def _intermediate_points(self, frame, from_, to, number_of_points):
        from_point = frame[from_]
        to_point = frame[to]
        diff = from_point - to_point
        step = diff / (number_of_points + 1)
        step_array = np.tile(step, number_of_points).reshape(-1, step.shape[0])

        intermediate_points = to_point + step_array * \
            np.arange(1, number_of_points + 1)[:, np.newaxis]
        return intermediate_points
