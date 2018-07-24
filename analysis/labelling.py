import numpy as np

from tracker import TrackVisualiser


class Labelling:

    @staticmethod
    def label_chunks(chunks, chunk_frames, video, processor):
        visualiser = TrackVisualiser()
        tracks = processor.chunks_to_tracks(chunks, chunk_frames)

        labels = {}

        i = 0
        while i < len(tracks):
            track = tracks[i]
            visualiser.draw_video_with_tracks(
                [track], video, track.frame_assigned[-1].astype(np.int), track.frame_assigned[0].astype(np.int))
            label = input('Label? (Scan, Cash, sTill, Moving, Other, q(skip))')
            if label in ['s', 'c', 'o', 'm', 't', 'q']:
                if label == 's':
                    label = 'scan'
                elif label == 'c':
                    label = 'cash'
                elif label == 'm':
                    label = 'moving'
                elif label == 't':
                    label = 'still'
                elif label == 'q':
                    i += 1
                    continue
                elif label == 'o':
                    label = 'other'
                labels[i] = label
                # If no valid action, show the video again.
                i += 1

        return labels
