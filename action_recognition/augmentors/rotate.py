import numpy as np


class Rotate:
    """Rotates point clouds for data augmentation.

    Parameters
    ----------
    number_of_added_points : int, optional
        The number of augmented points to add.

    """

    def __init__(self, number_of_added_points=2):
        self.number_of_added_points = number_of_added_points

    def augment(self, chunks, frames, labels, videos):
        """Rotates the inputted data randomly.

        Does not transform to origin, so take care if your data isn't
        already centered.

        Replicates frames, labels and videos for each added rotated
        cloud.

        Parameters
        ----------
        chunks : array-like, shape = [n_chunks, n_keypoints, 3]
        frames : array-like, shape = [n_chunks, n_frames, 1]
        labels : array-like, shape = [n_chunks, 1]
        videos : array-like, shape = [n_chunks, 1]

        Returns
        -------
        chunks : array-like
            shape = [n_chunks * (1 + number_of_added_points), n_keypoints, 3]
        frames : array-like
            shape = [n_chunks * (1 + number_of_added_points), n_frames, 1]
        labels : array-like
            shape = [n_chunks * (1 + number_of_added_points), 1]
        videos : array-like
            shape = [n_chunks * (1 + number_of_added_points), 1]

        """

        rotated_chunks = np.array([self._random_rotation(chunk)
                                   for chunk in chunks
                                   for i in range(self.number_of_added_points)])
        chunks = np.append(rotated_chunks, chunks, axis=0)

        frames = self._replicate(frames)
        labels = self._replicate(labels)
        videos = self._replicate(videos)

        return chunks, frames, labels, videos

    def _replicate(self, array):
        added_array = np.array([element
                                for element in array
                                for i in range(self.number_of_added_points)])
        return np.append(added_array, array, axis=0)

    def _random_rotation(self, chunk):
        rotated_chunk = np.copy(chunk)
        rotated_chunk[:, :, 2] = 0

        rotation_matrix = self._rotation_matrix()

        rotated_chunk = rotated_chunk @ rotation_matrix

        rotated_chunk[:, :, 2] = chunk[:, :, 2]

        return rotated_chunk

    def _rotation_matrix(self):
        #   Hand crafted parameters to get rotations that are significantly
        # different from the original, and to make sure that they do not
        # result in entirely flat people.
        tx = np.radians(np.random.choice([
            np.random.random_integers(low=30, high=70),
            np.random.random_integers(low=120, high=180)
        ]))
        ty = np.radians(np.random.choice([
            np.random.random_integers(low=30, high=70),
            np.random.random_integers(low=120, high=260)
        ]))
        tz = np.radians(np.random.random_integers(low=40, high=320))

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(tx), -np.sin(tx)],
            [0, np.sin(tx), np.cos(tx)]
        ])
        Ry = np.array([
            [np.cos(ty), 0, -np.sin(ty)],
            [0, 1, 0],
            [np.sin(ty), 0, np.cos(ty)]
        ])
        Rz = np.array([
            [np.cos(tz), -np.sin(tz), 0],
            [np.sin(tz), np.cos(tz), 0],
            [0, 0, 1]
        ])

        return Rz @ Ry @ Rx
