import numpy as np

from tracker import Person, Track


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

    def automatic_labelling(self, timestamps, frames_per_chunk):
        valid_timestamps = [t for t in timestamps
                            if t['end_frame'] - t['start_frame'] >= frames_per_chunk and
                            any(self._fits_in_timestamp(track, t) for track in self.tracks)]

        keypoints = self.tracks[0][0].keypoints
        chunks = np.zeros((len(valid_timestamps), frames_per_chunk, *keypoints.shape))
        frames = np.zeros((len(valid_timestamps), frames_per_chunk), dtype=np.int)
        labels = np.zeros(len(valid_timestamps), dtype=object)

        for i, timestamp in enumerate(valid_timestamps):
            # Only include the first track that fits the timestamp
            track = next(track for track in self.tracks
                         if self._fits_in_timestamp(track, timestamp))

            timestamp_length = timestamp['end_frame'] - timestamp['start_frame']
            middle_of_timestamp = timestamp['start_frame'] + timestamp_length / 2
            start_of_chunk = int(middle_of_timestamp - frames_per_chunk / 2)
            chunk, chunk_frames = track.chunk_from_frame(start_of_chunk, frames_per_chunk)

            chunks[i] = chunk
            frames[i] = chunk_frames
            labels[i] = timestamp['label']

        return chunks, frames, labels

    def _fits_in_timestamp(self, track, timestamp):
        track_start = track.frame_assigned[0]
        track_end = track.frame_assigned[-1]
        return track_start <= timestamp['start_frame'] and track_end >= timestamp['end_frame']
