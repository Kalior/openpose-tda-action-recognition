import numpy as np
import json
import cv2

from tracker import TrackVisualiser


class Labelling:

    def __init__(self):
        self.visualiser = TrackVisualiser()

    def keypress_valid(self, keypress):
        return keypress in self.valid_actions()

    def valid_actions(self):
        return ['s', 'c', 'o', 'm', 't', 'l', 'q', 'h', 'p']

    def valid_actions_string(self):
        return "(Scan, Cash, sTill, Moving, Lie, sHoplift, shoP, Other, quit)"

    def parse_keypress_to_label(self, keypress):
        if keypress == 's':
            label = 'scan'
        elif keypress == 'c':
            label = 'cash'
        elif keypress == 'm':
            label = 'moving'
        elif keypress == 't':
            label = 'still'
        elif keypress == 'l':
            label = 'lie'
        elif keypress == 'h':
            label = 'shoplift'
        elif keypress == 'p':
            label = 'shop'
        else:
            label = 'other'
        return label

    def label_chunks(self, chunks, chunk_frames, video, processor):
        tracks = processor.chunks_to_tracks(chunks, chunk_frames)

        labels = {}

        i = 0
        while i < len(tracks):
            track = tracks[i]
            self.visualiser.draw_video_with_tracks(
                [track], video, track.frame_assigned[-1].astype(np.int), track.frame_assigned[0].astype(np.int))
            label = input('Label? ' + self.valid_actions_string())
            if self.keypress_valid(label):
                if label == 'q':
                    i += 1
                    continue
                else:
                    label = self.parse_keypress_to_label(label)
                labels[i] = label
                i += 1
                # If no valid action, show the video again.

        return labels

    def pseudo_automatic_labelling(self, timestamps, frames_per_chunk, video, tracks):
        valid_timestamps = [t for t in timestamps
                            if t['end_frame'] - t['start_frame'] >= frames_per_chunk]

        keypoints = tracks[0][0].keypoints
        chunk_shape = (frames_per_chunk, *keypoints.shape)
        chunks, frames, labels, track_indicies = self._init_arrays(chunk_shape, frames_per_chunk)

        for timestamp in valid_timestamps:
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
        keypoints = tracks[0][0].keypoints
        chunks = np.zeros((len(json_labels), frames_per_chunk, *keypoints.shape))
        frames = np.zeros((len(json_labels), frames_per_chunk), dtype=np.int)
        labels = np.zeros(len(json_labels), dtype=object)

        for i, label in enumerate(json_labels):
            track_index = int(label['track_index'])
            track = tracks[track_index]

            start_frame = int(label['start_frame'])
            end_frame = int(label['end_frame'])

            if end_frame - start_frame != frames_per_chunk - 1:
                info.debug("Label {} did not have {} frames per chunk".format(
                    label, frames_per_chunk))

            chunk, chunk_frames = track.chunk_from_frame(start_frame, frames_per_chunk)

            chunks[i] = chunk
            frames[i] = chunk_frames
            labels[i] = label['label']

        return chunks, frames, labels

    def write_labels(self, chunks, chunk_frames, chunk_labels, track_indicies, labels_file):
        labels = []

        for i in range(len(chunks)):
            label = {
                'track_index': int(track_indicies[i]),
                'start_frame': int(chunk_frames[i][0]),
                'end_frame':   int(chunk_frames[i][-1]),
                'label': chunk_labels[i]
            }
            labels.append(label)

        with open(labels_file, 'w') as f:
            json.dump(labels, f)
