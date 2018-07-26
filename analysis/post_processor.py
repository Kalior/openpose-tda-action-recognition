import numpy as np
import scipy.signal

from tracker import Person, Track
from util import coco_connections


class PostProcessor:

    def __init__(self):
        self.tracks = []

    def chunks_to_tracks(self, chunks, chunk_frames):
        tracks = []
        for i, chunk in enumerate(chunks):
            t = self._create_track(
                chunk,
                chunk_frames[i], i)
            t.fill_missing_frames()
            tracks.append(t)

        return tracks

    def create_tracks(self, tracks_np, frames_np):
        self.tracks = [self._create_track(p, f, i)
                       for i, (p, f) in enumerate(zip(tracks_np, frames_np))]

    def _create_track(self, saved_path, path_frames, path_index):
        track = Track()
        for i in range(len(saved_path)):
            person = Person(saved_path[i])
            person.track_index = path_index
            track.add_person(person, path_frames[i])

        return track

    def post_process_tracks(self):
        self._combine_tracks(self.tracks)
        self.tracks = self._clean_tracks(self.tracks)
        self._fill_tracks(self.tracks)

    def _clean_tracks(self, tracks):
        return [track for track in tracks if len(track) > 15]

    def _fill_tracks(self, tracks):
        for track in tracks:
            track.fill_missing_keypoints()
            track.fill_missing_frames()

    def _combine_tracks(self, tracks):
        # This looks ugly since there is some concurrent modification going on.
        i = 0
        while i < len(tracks):
            have_removed = False
            end_track = tracks[i]
            j = i
            while j < len(tracks):
                start_track = tracks[j]
                if i != j and (end_track.overlaps(start_track) or
                               self._paths_nearby(end_track, start_track)):
                    end_track.combine(start_track)
                    end_track.fill_missing_keypoints()
                    tracks.remove(start_track)
                    have_removed = True
                else:
                    j += 1
            if not have_removed:
                i += 1

    def _paths_nearby(self, end_track, start_track):
        return ((end_track[-1].distance(start_track[0]) < 15 and
                 abs(end_track.frame_assigned[-1] - start_track.frame_assigned[0]) < 15))

    def chunk_tracks(self, frames_per_chunk, overlap, target_frames_per_chunk):
        number_of_keypoints = 18
        number_of_coordinates = 3
        chunks = np.empty((0, target_frames_per_chunk, number_of_keypoints, number_of_coordinates))
        chunk_frames = np.empty((0, target_frames_per_chunk), dtype=np.int)
        for i, track in enumerate(self.tracks):
            chunked, frames = track.divide_into_chunks(frames_per_chunk, overlap)
            if chunked.shape[0] > 0:
                chunked, frames = self._decrease_frames_per_chunk(
                    chunked, frames, target_frames_per_chunk)
                chunks = np.append(chunks, chunked, axis=0)
                chunk_frames = np.append(chunk_frames, frames, axis=0)

        return chunks, chunk_frames

    def _decrease_frames_per_chunk(self, chunked, chunk_frames, target_frames_per_chunk):
        indicies = np.arange(0, chunked.shape[1],
                             chunked.shape[1] / target_frames_per_chunk).astype(np.int)
        target_chunked = np.empty((chunked.shape[0], target_frames_per_chunk, *chunked.shape[2:]))
        target_frames = np.empty((chunk_frames.shape[0], target_frames_per_chunk), dtype=np.int)
        for i, chunk in enumerate(chunked):
            target_chunked[i] = chunk[indicies]
            target_frames[i] = chunk_frames[i, indicies]
        return target_chunked, target_frames

    def filter_moving_chunks(self, chunks, chunk_frames):
        number_of_keypoints = 18
        number_of_coordinates = 3
        filtered_chunks = np.empty((0, *chunks.shape[1:]))
        filtered_frames = np.empty((0, chunk_frames.shape[1]))

        for i, chunk in enumerate(chunks):
            position = [820, 350]
            mean = chunk[~np.all(chunk == 0, axis=2)].mean(axis=0)
            if np.linalg.norm(mean[:2] - position) < 100:
                filtered_chunks = np.append(filtered_chunks, [chunk], axis=0)
                filtered_frames = np.append(filtered_frames, [chunk_frames[i]], axis=0)

        return filtered_chunks, filtered_frames

    def flatten_chunks(self, chunks, selected_keypoints):
        data = np.array([chunk[:, selected_keypoints, :2].flatten()
                         for chunk in chunks])

        return data

    def flatten_chunks_3D(self, chunks, selected_keypoints, connect_keypoints):
        data = np.array([self._chunk_to_3D(chunk, selected_keypoints, connect_keypoints)
                         for chunk in chunks])

        return data

    def _chunk_to_3D(self, chunk, selected_keypoints, connect_keypoints):
        number_of_frames = chunk.shape[0]
        flat_chunk = np.vstack(self._connect_keypoints(
            chunk[:, selected_keypoints, :2], connect_keypoints, 2))
        third_dimension = np.repeat(np.arange(0, number_of_frames), len(
            selected_keypoints) + len(connect_keypoints) * 2)
        return np.c_[flat_chunk, third_dimension]

    def translate_chunks_to_origin(self, chunks):
        translated_chunks = np.copy(chunks)

        for i, chunk in enumerate(translated_chunks):
            self._normalise_chunk(chunk)

        return translated_chunks

    def _normalise_chunk(self, chunk):
        # Don't take unidentified keypoints into account:
        mean = chunk[~np.all(chunk == 0, axis=2)].mean(axis=0)

        chunk[~np.all(chunk == 0, axis=2)] -= mean

    def translate_chunks_to_origin_by_keypoint(self, chunks):
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

    def _connect_keypoints(self, chunk, connect_keypoints, number_of_points=3):
        # return chunk
        new_number_of_keypoints = chunk.shape[1] + len(connect_keypoints) * number_of_points
        connected_chunk = np.zeros((chunk.shape[0], new_number_of_keypoints, chunk.shape[2]))
        connected_chunk[:, :chunk.shape[1]] = chunk
        for i, frame in enumerate(chunk):
            for j, (from_, to) in enumerate(connect_keypoints):
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

    def smooth_chunks(self, chunks):
        smooth = np.copy(chunks)
        for chunk in smooth:
            self._smooth_chunk(chunk)
        return smooth

    def _smooth_chunk(self, chunk):
        window_length = int(chunk.shape[0] / 4)
        for i in range(chunk.shape[1]):
            for j in range(2):
                chunk[:, i, j] = scipy.signal.savgol_filter(chunk[:, i, j], window_length, 3)
