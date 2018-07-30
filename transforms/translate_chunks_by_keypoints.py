import numpy as np


class TranslateChunksByKeypoints:

    def transform(self, chunks):
        translated_chunks = np.copy(chunks)

        for i, chunk in enumerate(translated_chunks):
            self._translate_by_keypoint(chunk)

        return translated_chunks

    def _translate_by_keypoint(self, chunk):
        for i in range(chunk.shape[1]):
            # Don't take unidentified keypoints into account:
            keypoints = chunk[:, i][~np.all(chunk[:, i] == 0, axis=1)]
            if keypoints.shape[0] != 0:
                keypoint_mean = keypoints.mean(axis=0)
                chunk[:, i][~np.all(chunk[:, i] == 0, axis=1)] -= keypoint_mean
