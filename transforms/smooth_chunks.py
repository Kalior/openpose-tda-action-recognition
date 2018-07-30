import scipy.signal
import numpy as np


class SmoothChunks:

    def transform(self, chunks):
        smooth = np.copy(chunks)
        for chunk in smooth:
            self._smooth_chunk(chunk)
        return smooth

    def _smooth_chunk(self, chunk):
        window_length = int(chunk.shape[0] / 4)
        for i in range(chunk.shape[1]):
            for j in range(2):
                chunk[:, i, j] = scipy.signal.savgol_filter(chunk[:, i, j], window_length, 3)
