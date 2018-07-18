import cv2
import numpy as np
from time import sleep

from tracker import TrackVisualiser, Track, Person


class MapperVisualiser:

    def __init__(self, chunks, chunk_frames, translated_chunks, frames_per_chunk, selected_keypoints, video):
        self.chunks = chunks
        self.chunk_frames = chunk_frames
        self.translated_chunks = translated_chunks
        self.frames_per_chunk = frames_per_chunk
        self.selected_keypoints = selected_keypoints
        self.video = video

    def visualise(self, graph, labels):
        capture = cv2.VideoCapture(self.video)

        nodes = graph['nodes']
        for name, node in nodes.items():
            self._draw_node(capture, name, node, labels)
            sleep(1)

    def _draw_node(self, capture, name, node, labels):
        visualiser = TrackVisualiser()

        average_frames = self._draw_average_shape(capture, name, node, labels, visualiser)
        self._draw_every_chunk(capture, name, node, labels, visualiser, average_frames)

    def _draw_every_chunk(self, capture, name, node, labels, visualiser, average_frames):
        for point in node:
            chunk_index = labels[point][0]
            person_index = labels[point][1]

            start_frame = labels[point][2]
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            original_chunk = self.chunks[person_index][chunk_index]
            translated_chunk = self.translated_chunks[person_index][chunk_index]

            self._draw_chunk(capture, name, original_chunk,
                             translated_chunk, start_frame, visualiser, average_frames)

    def _draw_chunk(self, capture, name, chunk, translated_chunk, start_frame, visualiser, average_frames):
        track = self._chunk_to_track(chunk, start_frame)

        translated_track = self._chunk_to_track(translated_chunk, start_frame)

        for i in range(self.frames_per_chunk):
            success, original_image = capture.read()
            visualiser.draw_tracks([track], original_image, i + start_frame)
            visualiser.draw_frame_number(original_image, i + start_frame)

            translated_image = self._draw_translated_track(
                translated_track, i, start_frame, visualiser)

            visualiser.draw_text(translated_image, name, position=(20, 450))
            visualiser.draw_text(original_image, name, position=(1400, 50))
            cv2.imshow("output", original_image)
            cv2.imshow("translated_person", translated_image)
            cv2.imshow("average person", average_frames[i])
            cv2.waitKey(1)

    def _draw_average_shape(self, capture, name, node, labels, visualiser):
        tracks = []
        for point in node:
            chunk_index = labels[point][0]
            person_index = labels[point][1]
            start_frame = labels[point][2]

            chunk = self.translated_chunks[person_index][chunk_index]
            track = self._chunk_to_track(chunk, start_frame)
            tracks.append((start_frame, track))

        frames = []
        opacity = 1 / len(tracks) + 0.05
        for i in range(self.frames_per_chunk):
            blank_image = np.zeros((500, 500, 3), np.uint8)
            for start_frame, track in tracks:
                overlay = blank_image.copy()
                visualiser.draw_people([track], overlay, i + start_frame)
                cv2.addWeighted(overlay, opacity, blank_image, 1 - opacity, 0, blank_image)
            visualiser.draw_text(blank_image, name, position=(20, 50))

            frames.append(blank_image)

        return frames

    def _chunk_to_track(self, chunk, start_frame):
        track = Track()
        for i, p in enumerate(chunk):
            track.add_person(Person(p), i + start_frame)

        return track

    def chunk_to_video_pose(self, chunk, out_file, start_frame):
        translated_track = self._chunk_to_track(chunk, start_frame)

        writer, frame_width, frame_height = self._create_writer(out_file)
        visualiser = TrackVisualiser()

        for i in range(self.frames_per_chunk):
            translated_image = self._draw_translated_track(
                translated_track, i, start_frame, visualiser)
            translated_image = cv2.resize(translated_image, (frame_width, frame_height))
            writer.write(translated_image)

        writer.release()

    def _create_writer(self, out_file):
        frame_width = 100
        frame_height = 100
        fps = 10
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        writer = cv2.VideoWriter(out_file, fourcc, fps, (frame_width, frame_height))
        return writer, frame_width, frame_height

    def _draw_translated_track(self, translated_track, i, start_frame, visualiser):
        blank_image = np.zeros((500, 500, 3), np.uint8)
        visualiser.draw_frame_number(blank_image, i + start_frame)
        visualiser.draw_people([translated_track], blank_image, i + start_frame)

        return blank_image

    def chunk_to_video_scene(self, chunk, out_file, start_frame):
        mean = chunk[~np.all(chunk == 0, axis=2)].mean(axis=0)
        crop_size = 500
        start_y = int(mean[0] - crop_size / 2)
        start_x = int(mean[1] - crop_size / 2)

        capture = cv2.VideoCapture(self.video)
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        track = self._chunk_to_track(chunk, start_frame)
        visualiser = TrackVisualiser()

        writer, frame_width, frame_height = self._create_writer(out_file)

        for i in range(self.frames_per_chunk):
            sucess, image = capture.read()
            cropped_image = self._draw_track_scene(
                track, i, start_frame, visualiser, image, start_x, start_y, crop_size)
            scaled_image = cv2.resize(cropped_image, (frame_width, frame_height))
            writer.write(scaled_image)

        writer.release()

    def _draw_track_scene(self, track, i, start_frame, visualiser,
                          original_image, start_x, start_y, crop_size):
        visualiser.draw_frame_number(original_image, i + start_frame)
        visualiser.draw_people([track], original_image, i + start_frame, offset_person=False)

        return original_image[start_x:(start_x + crop_size), start_y:(start_y + crop_size)]
