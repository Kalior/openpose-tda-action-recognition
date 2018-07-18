import numpy as np

from tracker import Person, Track


class PostProcessor:

    def __init__(self):
        self.tracks = []

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

    def translate_tracks_to_origin(self):
        for track in self.tracks:
            track.translate_to_origin()

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
                               self._paths_close(end_track, start_track)):
                    end_track.combine(start_track)
                    end_track.fill_missing_keypoints()
                    tracks.remove(start_track)
                    have_removed = True
                else:
                    j += 1
            if not have_removed:
                i += 1

    def _paths_close(self, end_track, start_track):
        return ((end_track[-1].distance(start_track[0]) < 15 and
                 abs(end_track.frame_assigned[-1] - start_track.frame_assigned[0]) < 15))

    def chunk_tracks(self, frames_per_chunk, overlap):
        chunks = np.empty(len(self.tracks), dtype=object)
        chunk_frames = np.empty(len(self.tracks), dtype=object)
        for i, track in enumerate(self.tracks):
            chunked, frames = track.divide_into_chunks(frames_per_chunk, overlap)
            chunks[i] = chunked
            chunk_frames[i] = frames

        return chunks, chunk_frames

    def filter_moving_chunks(self, chunks, chunk_frames):
        filtered_chunks = np.empty(chunks.shape[0], dtype=object)
        filtered_frames = np.empty(chunks.shape[0], dtype=object)
        for i, track in enumerate(chunks):
            filtered_track = []
            filtered_track_frames = []
            for j, chunk in enumerate(track):
                position = [820, 350]
                mean = chunk[~np.all(chunk == 0, axis=2)].mean(axis=0)
                if np.linalg.norm(mean[:2] - position) < 100:
                    filtered_track.append(chunk)
                    filtered_track_frames.append(chunk_frames[i][j])
            filtered_chunks[i] = np.array(filtered_track)
            filtered_frames[i] = np.array(filtered_track_frames)

        return filtered_chunks, filtered_frames

    def chunks_to_tracks(self, chunks, chunk_frames):
        tracks = []
        for i, track in enumerate(chunks):
            for j, chunk in enumerate(track):
                t = self._create_track(
                    chunk,
                    [chunk_frames[i][j] + k for k in range(len(chunk))],
                    i * len(track) + j)
                tracks.append(t)

        return tracks
