class Labelling:

    @staticmethod
    def label_chunks(chunks, chunk_frames, video, processor):
        visualiser = TrackVisualiser()
        tracks = processor.chunks_to_tracks(chunks, chunk_frames)

        labels = {}

        for i, track in enumerate(tracks):
            visualiser.draw_video_with_tracks(
                [track], video, track.frame_assigned[-1].astype(np.int), track.frame_assigned[0].astype(np.int))
            label = input('Label? (Scan, Cash, sTill, Moving, Other)')
            if label in ['s', 'c', 'o', 'm', 't']:
                if label == 's':
                    label = 'scan'
                elif label == 'c':
                    label = 'cash'
                elif label == 'm':
                    label = 'moving'
                elif label == 't':
                    label = 'still'
                else:
                    label = 'other'
            else:
                label = 'other'
            labels[i] = label
            print()

        return labels
