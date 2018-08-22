import numpy as np
import json
import cv2
import logging

from ..tracker import TrackVisualiser


class Labelling:
    """Help class for labelling datasets.

    Contains a dict with the valid actions as well as some
    helper functions which display chunks to the user and prompts
    either for a label or for verification of a label.

    """

    def __init__(self):
        self.visualiser = TrackVisualiser()
        self.actions = {
            's': ['scan', 'Scan'],
            'c': ['cash', 'Cash'],
            't': ['still', 'sTill'],
            'm': ['moving', 'Moving'],
            'l': ['lie', 'Lie'],
            'p': ['shop', 'shoP'],
            'h': ['shoplift', 'sHoplit'],
            'o': ['other', 'Other']
        }

    def keypress_valid(self, keypress):
        """Check for if a given keypress is valid.

        Parameters
        ----------
        keypress : char

        Returns
        -------
        valid : boolean, True if keypress is contained in valid_actions()

        """
        return keypress in self.valid_actions()

    def valid_actions(self):
        """Returns a list of the possible valid characters.

        Returns
        -------
        valid_actions : list
            contains characters corresponding to the valid actions.

        """
        return list(self.actions.keys())

    def valid_actions_string(self):
        """Returns the string corresponding to the prompt of valid actions.

        Returns
        -------
        valid_actions_string : string
            Corresponds to the prompt of valid actions.
        """
        return "(" + ", ".join([v[1] for v in self.actions.values()]) + ', quit)'

    def parse_keypress_to_label(self, keypress):
        """Parses a character to a label

        Parameters
        ----------
        keypress : char

        Returns
        -------
        label : string of the corresponding label to the keypress
        """
        if keypress in self.actions:
            return self.actions[keypress][0]
        else:
            return 'other'

    def label_chunks(self, chunks, chunk_frames, video, processor):
        """Function for manual labelling of chunks.

        Will prompt every passed chunk for a label.

        Parameters
        ----------
        chunks : numpy array shape = [number_of_frames, number_of_keypoints, 3]
        chunk_frames : numpy array shape = [number_of_frames]
        video : str, path to a video file from where the chunks were taken
        processor : analysis.PostProcessor object

        Returns
        -------
        labels : dict with label[index] = label per labelled chunk.

        """

        tracks = processor.chunks_to_tracks(chunks, chunk_frames)

        labels = {}

        i = 0
        while i < len(tracks):
            track = tracks[i]
            self.visualiser.draw_video_with_tracks(
                [track], video, track.frame_assigned[-1].astype(np.int), track.frame_assigned[0].astype(np.int))
            keypress = input('Label? ' + self.valid_actions_string())
            if self.keypress_valid(keypress):
                label = self.parse_keypress_to_label(label)
                labels[i] = label
                i += 1
            elif keypress == 'q':
                i += 1
                continue
                # If no valid action, show the video again.

        return labels

    def pseudo_automatic_labelling(self, timestamps, frames_per_chunk, video, tracks):
        """Makes use of timestamps from recording-time to ease labelling.

        Only prompts for if the displayed video should be discarded or accepted
        with the already given label.  Also provides option to skip forward by
        a fourth of the length of each chunk in order to better align actions.

        Parameters
        ----------
        timestamps : list of dicts
            each dict must contain a 'start_time', 'end_time', and 'label' for
            each timestamp.
        frames_per_chunk : int, corresponding to how long each chunk is.
        video : str, path to the video corresponding to the timestamps.
        tracks : list of tracker.Track
            the tracks from video, will be divided up into chunks of length
            frames_per_chunk.

        Returns
        -------
        chunks : numpy.array of the labelled chunks
        frames : numpy.array of the frames for the labelled chunks
        labels : numpy.array of the labels for each chunk
        indicies : numpy.array of the index of the track for every chunk.
            needed for reproducability.
        """
        keypoints = tracks[0][0].keypoints
        chunk_shape = (frames_per_chunk, *keypoints.shape)
        chunks, frames, labels, track_indicies = self._init_arrays(chunk_shape, frames_per_chunk)

        for timestamp in timestamps:
            # Only include the first track that fits the timestamp
            for i, track in enumerate(tracks):
                track_arrays = self._pseudo_automatic_labelling(
                    timestamp, track, i, frames_per_chunk, chunk_shape, video)
                chunks = np.append(chunks, track_arrays[0], axis=0)
                frames = np.append(frames, track_arrays[1], axis=0)
                labels = np.append(labels, track_arrays[2], axis=0)
                track_indicies = np.append(track_indicies, track_arrays[3], axis=0)

        return chunks, frames, labels, track_indicies

    def _init_arrays(self, chunk_shape, frames_per_chunk):
        chunks = np.zeros((0, *chunk_shape))
        frames = np.zeros((0, frames_per_chunk), dtype=np.int)
        labels = np.zeros(0, dtype=object)
        track_indicies = np.zeros(0, dtype=np.int)
        return chunks, frames, labels, track_indicies

    def _pseudo_automatic_labelling(self, timestamp, track, track_index, frames_per_chunk, chunk_shape, video):
        capture = cv2.VideoCapture(video)
        fps = capture.get(cv2.CAP_PROP_FPS)

        chunks, frames, labels, track_indicies = self._init_arrays(chunk_shape, frames_per_chunk)

        start_frame = int(timestamp['start_time'] * fps)
        end_frame = start_frame + frames_per_chunk

        stamp_end = int(timestamp['end_time'] * fps)

        while end_frame <= stamp_end:
            if not self._fits_in_timestamp(track, start_frame, end_frame):
                start_frame += int(frames_per_chunk / 2)
                end_frame = start_frame + frames_per_chunk
                continue

            self.visualiser.draw_video_with_tracks([track], video, end_frame, start_frame)

            ok = input("Labelling as {}, ok? (y/n/s)".format(timestamp['label']))
            if ok == 'y' or ok == '':
                chunk, chunk_frames = track.chunk_from_frame(start_frame, frames_per_chunk)

                chunks = np.append(chunks, [chunk], axis=0)
                frames = np.append(frames, [chunk_frames], axis=0)
                labels = np.append(labels, [timestamp['label']], axis=0)
                track_indicies = np.append(track_indicies, [track_index], axis=0)

                start_frame += frames_per_chunk
            elif ok == 's':
                start_frame += int(frames_per_chunk / 4)
            else:
                start_frame += frames_per_chunk

            end_frame = start_frame + frames_per_chunk

        return chunks, frames, labels, track_indicies

    def _fits_in_timestamp(self, track, start_frame, end_frame):
        track_start = track.frame_assigned[0]
        track_end = track.frame_assigned[-1]
        return track_start <= start_frame and track_end >= end_frame

    def parse_labels(self, json_labels, frames_per_chunk, tracks):
        """Function for parsing saved labels from json to labelled chuns.

        Parameters
        ----------
        json_labels : list of dicts
            each dict must contain a 'track_index', 'start_frame', 'end_frame',
            and 'label' value corresponding to each labelled chunk.
        frames_per_chunk : int, length of each chunk.
        tracks : list of tracker.Track
            Must be the same list of tracks from which the original json_labels
            where created.

        Returns
        -------
        chunks : numpy.array of the labelled chunks
        frames : numpy.array of the frames for the labelled chunks
        labels : numpy.array of the labels for each chunk

        """
        keypoints = tracks[0][0].keypoints
        number_of_labels = len(json_labels)
        chunks = np.zeros(number_of_labels, dtype=object)
        frames = np.zeros(number_of_labels, dtype=object)
        labels = np.zeros(number_of_labels, dtype=object)

        for i, label in enumerate(json_labels):
            track_index = int(label['track_index'])
            track = tracks[track_index]

            start_frame = int(label['start_frame'])
            end_frame = int(label['end_frame'])

            if end_frame - start_frame != frames_per_chunk - 1:
                logging.info("Label {} did not have {} frames per chunk".format(
                    label, frames_per_chunk))

            chunk, chunk_frames = track.chunk_from_frame(start_frame, frames_per_chunk)
            if chunk.shape[0] < frames_per_chunk:
                logging.warn("Label {} does not allow for a chunk with size {}".format(
                    label, frames_per_chunk))

            chunks[i] = chunk
            frames[i] = chunk_frames
            labels[i] = label['label']

        return chunks, frames, labels

    def write_labels(self, chunks, chunk_frames, chunk_labels, track_indicies, labels_file):
        """Writes a set of labels to a json file, for later reproducibility.

        Parameters
        ----------
        chunks : numpy.array of labelled chunks.
        chunk_frames : numpy.array of the frames for the labelled chunks.
        chunk_labels : numpy.array of the labels for each chunk.
        track_indicies : numpy.array of the index of the track for every chunk.
        labels_file : str, path to where the labels are to be written.

        """
        labels = []

        for i in range(len(chunks)):
            label = {
                'track_index': int(track_indicies[i]),
                'start_frame': int(chunk_frames[i][0]),
                'end_frame': int(chunk_frames[i][-1]),
                'label': chunk_labels[i]
            }
            labels.append(label)

        with open(labels_file, 'w') as f:
            json.dump(labels, f)

    def manual_labelling(self, video, processor, target_frames_per_chunk, overlap_percentage, seconds_per_chunk):
        """Divides tracks into chunks with the specified overlap, and prompts the user for a label for each.

        Parameters
        ----------
        video : str, path to the video for which the tracks where created.
        processor : Processor object
            Requires that create_tracks(...) has been called on the processor
            before this function call.
        target_frames_per_chunk : int
            Determines how long each chunk is in the output.
        overlap_percentage : float
            The percentage (from 0 to 1) of overlap between suggested chunks.
        seconds per chunk : float
            The number of seconds that each chunk should contain.

        Returns
        -------
        chunks : numpy.array of the labelled chunks
        frames : numpy.array of the frames for the labelled chunks
        labels : numpy.array of the labels for each chunk
        indicies : numpy.array of the index of the track for every chunk.
            needed for reproducability.
        """
        capture = cv2.VideoCapture(video)
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames_per_chunk_for_seconds = int(seconds_per_chunk * fps)
        logging.debug("Frames per chunks to get same #seconds: {}".format(
            frames_per_chunk_for_seconds))
        overlap = int(frames_per_chunk_for_seconds * overlap_percentage)

        logging.info("Chunking tracks.")
        chunks, frames, track_indicies = processor.chunk_tracks(
            frames_per_chunk_for_seconds, overlap, target_frames_per_chunk)
        logging.info("Identified {} chunks".format(chunks.shape[0]))

        labels = self.label_chunks(chunks, frames, video, processor)

        chunks = chunks[np.array(list(labels.keys()), dtype=np.int)]
        frames = frames[np.array(list(labels.keys()), dtype=np.int)]
        track_indicies = track_indicies[np.array(list(labels.keys()), dtype=np.int)]

        return chunks, frames, list(labels.values()), track_indicies
