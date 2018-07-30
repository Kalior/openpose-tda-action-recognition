import numpy as np


class TranslateChunks:

    def transform(self, chunks):
        translated_chunks = np.copy(chunks)

        for i, chunk in enumerate(translated_chunks):
            self._normalise_chunk(chunk)

        return translated_chunks

    def _normalise_chunk(self, chunk):
        # Don't take unidentified keypoints into account:
        mean = chunk[~np.all(chunk == 0, axis=2)].mean(axis=0)

        chunk[~np.all(chunk == 0, axis=2)] -= mean
