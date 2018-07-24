import cv2
import numpy as np
from time import sleep

from tracker import TrackVisualiser, Track, Person


class ChunkVisualiser:

    def __init__(self, chunks, chunk_frames, translated_chunks):
        self.chunks = chunks
        self.chunk_frames = chunk_frames
        self.translated_chunks = translated_chunks

    def visualise(self, videos, graph):
        nodes = graph['nodes']
        for name, node in nodes.items():
            self.draw_node(videos, name, node)
            sleep(1)

    def visualise_averages(self, nodes):
        visualiser = TrackVisualiser()
        all_average_frames = []
        for name, node in nodes.items():
            average_frames = self._draw_average_shape(name, node, visualiser)
            all_average_frames.append(average_frames)

        while True:
            for i in range(20):
                for j, average_frames in enumerate(all_average_frames):
                    smaller_average = cv2.resize(average_frames[i], (0, 0), fx=0.5, fy=0.5)
                    cv2.imshow("average person" + str(j), smaller_average)
                    cv2.waitKey(15)

    def draw_node(self, videos, name, node):
        visualiser = TrackVisualiser()

        average_frames = self._draw_average_shape(name, node, visualiser)
        self._draw_every_chunk(videos, name, node, visualiser, average_frames)

    def _draw_every_chunk(self, videos, name, node, visualiser, average_frames):
        for point in node:
            capture = cv2.VideoCapture(videos[point])
            start_frame = self.chunk_frames[point]
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            original_chunk = self.chunks[point]
            translated_chunk = self.translated_chunks[point]

            self._draw_chunk(capture, name, original_chunk,
                             translated_chunk, start_frame, visualiser, average_frames)

    def _draw_chunk(self, capture, name, chunk, translated_chunk, start_frame, visualiser, average_frames):
        track = self._chunk_to_track(chunk, start_frame)

        translated_track = self._chunk_to_track(translated_chunk, start_frame)

        for i in range(len(chunk)):
            success, original_image = capture.read()
            visualiser.draw_tracks([track], original_image, i + start_frame)
            visualiser.draw_frame_number(original_image, i + start_frame)

            translated_image = self._draw_translated_track(
                translated_track, i, start_frame, visualiser)

            visualiser.draw_text(translated_image, name, position=(20, 450))
            visualiser.draw_text(original_image, name, position=(1400, 50))

            smaller_original = cv2.resize(original_image, (0, 0), fx=0.5, fy=0.5)
            smaller_translated = cv2.resize(translated_image, (0, 0), fx=0.5, fy=0.5)
            smaller_average = cv2.resize(average_frames[i], (0, 0), fx=0.5, fy=0.5)
            cv2.imshow("output", smaller_original)
            cv2.imshow("translated_person", smaller_translated)
            cv2.imshow("average person", smaller_average)
            cv2.waitKey(1)

    def _draw_average_shape(self, name, node, visualiser):
        tracks = []
        for point in node:
            start_frame = self.chunk_frames[point]

            chunk = self.translated_chunks[point]
            track = self._chunk_to_track(chunk, start_frame)
            tracks.append((start_frame, track))

        frames = []
        opacity = 1 / len(tracks) + 0.05
        for i in range(len(tracks[0][1])):
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

        for i in range(len(chunk)):
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

    def chunk_to_video_scene(self, video, chunk, out_file, start_frame, label):
        capture = cv2.VideoCapture(video)

        mean = chunk[~np.all(chunk == 0, axis=2)].mean(axis=0)
        crop_size = int(min(500,  min(capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                                      capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        start_y = max(0, int(mean[0] - crop_size / 2))
        start_x = max(0, int(mean[1] - crop_size / 2))

        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        track = self._chunk_to_track(chunk, start_frame)
        visualiser = TrackVisualiser()

        writer, frame_width, frame_height = self._create_writer(out_file)

        for i in range(len(chunk)):
            sucess, image = capture.read()
            cropped_image = self._draw_track_scene(
                track, i, start_frame, visualiser, image, start_x, start_y, crop_size, label)
            scaled_image = cv2.resize(cropped_image, (frame_width, frame_height))
            writer.write(scaled_image)

        writer.release()

    def _draw_track_scene(self, track, i, start_frame, visualiser,
                          original_image, start_x, start_y, crop_size, label):
        visualiser.draw_people([track], original_image, i + start_frame, offset_person=False)
        cropped_image = original_image[
            start_x:(start_x + crop_size), start_y:(start_y + crop_size)]
        visualiser.draw_frame_number(cropped_image, i + start_frame)
        visualiser.draw_text(cropped_image, label, position=(20, 450), color=(255, 255, 0))
        return cropped_image