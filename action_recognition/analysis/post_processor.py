import numpy as np

from ..tracker import Person, Track


class PostProcessor:
    """Encapsulates post processing functionality for tracks.

    Some examples are dividing tracks into chunks, removing short tracks, and
    merging/removing overlapping tracks.

    """

    def __init__(self):
        self.tracks = []

    def chunks_to_tracks(self, chunks, chunk_frames):
        """Transforms chunks into Track object.

        This conversion is convenient since the tracks have visualisation
        procedures.

        Parameters
        ----------
        chunks : array-like, shape = [n_chunks, frames_per_chunk, n_keypoints, 3]
            The chunks to convert into Tracks. Each chunk becomes a Track.
        chunk_frames : array-like, shape = [n_chunks, frames_per_chunk, 1]
            The frame numbers corresponding to the chunks.

        Returns
        -------
        tracks : array-like of Track objects.

        """
        tracks = []
        for i, chunk in enumerate(chunks):
            t = self._create_track(
                chunk,
                chunk_frames[i], i)
            t.fill_missing_frames()
            tracks.append(t)

        return tracks

    def create_tracks(self, tracks_np, frames_np):
        """Creates tracks from numpy format of tracks, as outputted from tracker.video().

        Parameters
        ----------
        tracks_np : array-like,
            numpy format of Track objects.
        frames_np : array-like
            numpy format of frame numbers for the Track objects.

        """
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
        """Combines, removes short tracks, and fills in missing parts of trakcs.

        Operates on the tracks contained in the class.

        """
        self._combine_tracks(self.tracks)
        self.tracks = self._clean_tracks(self.tracks)
        self._fill_tracks(self.tracks)

    def _clean_tracks(self, tracks):
        return [track for track in tracks if len(track) > 15]

    def _fill_tracks(self, tracks):
        for track in tracks:
            track.reset_keypoints()
            track.fill_missing_keypoints(fill_type='diff')
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
        """Divides the tracks into chunks the specified parameters.

        Parameters
        ----------
        frames_per_chunk : int
            The number of frames per chunk to divide into to maintain the
            same length of action across different videos.
        overlap : int
            The overlap allowed between chunks.
        target_frames_per_chunk : int
            The number of frames per chunk in the output, used to maintain the
            same dimensionality across videos.

        Returns
        -------
        chunks : array-like, shape = [n_chunks, target_frames_per_chunk, n_keypoints, 3]
            The chunks that the tracks are divided into.
        chunk_frames : array-like, shape = [n_chunks, target_frames_per_chunk, 1]
            The frame numbers corresponding to the chunks, used for visualisation.
        track_indicies : array-like, shape = [n_chunks, 1]
            The track index for each chunks, necessary for reproducibility.

        """
        number_of_keypoints = 18
        number_of_coordinates = 3
        chunks = np.empty((0, target_frames_per_chunk, number_of_keypoints, number_of_coordinates))
        chunk_frames = np.empty((0, target_frames_per_chunk), dtype=np.int)
        track_indicies = np.empty((0,), dtype=np.int)

        for i, track in enumerate(self.tracks):
            chunked, frames = track.divide_into_chunks(frames_per_chunk, overlap)
            if chunked.shape[0] > 0:
                chunked, frames = self._decrease_frames_per_chunk(
                    chunked, frames, target_frames_per_chunk)

                chunks = np.append(chunks, chunked, axis=0)
                chunk_frames = np.append(chunk_frames, frames, axis=0)
                indicies = np.array([i] * len(chunked))
                track_indicies = np.append(track_indicies, indicies, axis=0)

        return chunks, chunk_frames, track_indicies

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
        """Filters out chunks that are close to a specific position.

        *Do not use*, was used for the original cachier video, and needs to
        be adapted to work for a general case.

        Parameters
        ----------
        chunks : array-like, shape = [n_chunks, frames_per_chunk, n_keypoints, 3]
        chunks_frames : array-like, shape = [n_chunks, frames_per_chunk, 1]

        Returns
        -------
        chunks : array-like, shape = [n_still_chunks, frames_per_chunk, n_keypoints, 3]
        chunks_frames : array-like, shape = [n_still_chunks, frames_per_chunk, 1]

        """
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
