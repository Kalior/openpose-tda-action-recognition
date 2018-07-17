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

    def chunk_tracks(self):
        frames_per_chunk = 10
        overlap = 0

        # Some python magic for getting the structure right here...
        chunks_and_frames = [t for t in zip(*[track.divide_into_chunks(frames_per_chunk, overlap)
                                              for track in self.tracks])]
        chunks = np.array(chunks_and_frames[0])
        chunk_frames = np.array(chunks_and_frames[1])

        return chunks, chunk_frames
