import numpy as np

from tracker import TrackVisualiser


class Labelling:

    @staticmethod
    def label_chunks(chunks, chunk_frames, video, processor):
        visualiser = TrackVisualiser()
        tracks = processor.chunks_to_tracks(chunks, chunk_frames)

        labels = {}

        for i, track in enumerate(tracks):
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
                    continue
                else:
                    label = 'other'
            else:
                label = 'other'
            labels[i] = label

        return labels
