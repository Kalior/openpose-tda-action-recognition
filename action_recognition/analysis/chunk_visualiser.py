import cv2
import numpy as np
from time import sleep

from ..tracker import TrackVisualiser, Track, Person


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

    def visualise_averages(self, nodes, forever=False):
        visualiser = TrackVisualiser()
        all_average_frames = []
        for name, node in nodes.items():
            average_frames = self._draw_average_shape(name, node, visualiser)
            all_average_frames.append(average_frames)

        if forever:
            iteration_length = 99999
        else:
            iteration_length = 10
        for _ in range(iteration_length):
            for i in range(len(all_average_frames[0])):
                for j, average_frames in enumerate(all_average_frames):
                    smaller_average = cv2.resize(average_frames[i], (0, 0), fx=0.5, fy=0.5)
                    cv2.imshow("average person {}".format(j), smaller_average)
                    cv2.waitKey(15)

    def draw_node(self, videos, name, node):
        visualiser = TrackVisualiser()

        average_frames = self._draw_average_shape(name, node, visualiser)
        self._draw_every_chunk(videos, name, node, visualiser, average_frames)

    def _draw_every_chunk(self, videos, name, node, visualiser, average_frames):
        for point in node:
            capture = cv2.VideoCapture(videos[point])
            frames = self.chunk_frames[point]
            capture.set(cv2.CAP_PROP_POS_FRAMES, frames[0])

            original_chunk = self.chunks[point]
            translated_chunk = self.translated_chunks[point]

            self._draw_chunk(capture, name, original_chunk,
                             translated_chunk, frames, visualiser, average_frames)

    def _draw_chunk(self, capture, name, chunk, translated_chunk, frames, visualiser, average_frames):
        track = self._chunk_to_track(chunk, frames)

        translated_track = self._chunk_to_track(translated_chunk, frames)

        for i in range(len(chunk)):
            success, original_image = capture.read()
            visualiser.draw_tracks([track], original_image, frames[i])
            visualiser.draw_frame_number(original_image, frames[i])

            translated_image = self._draw_translated_track(
                translated_track, frames[i], visualiser)

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
            frames = self.chunk_frames[point]

            chunk = self.translated_chunks[point]
            track = self._chunk_to_track(chunk, frames)
            tracks.append((frames[0], track))

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

    def _chunk_to_track(self, chunk, frames):
        track = Track()
        for i, p in enumerate(chunk):
            track.add_person(Person(p), frames[i])

        track.fill_missing_frames()

        return track

    def chunk_to_video_pose(self, chunk, out_file, frames):
        translated_track = self._chunk_to_track(chunk, frames)

        writer, frame_width, frame_height = self._create_writer(out_file)
        visualiser = TrackVisualiser()

        for i in range(len(chunk)):
            translated_image = self._draw_translated_track(
                translated_track, i, frames, visualiser)
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

    def _draw_translated_track(self, translated_track, frame, visualiser):
        blank_image = np.zeros((500, 500, 3), np.uint8)
        visualiser.draw_frame_number(blank_image, frame)
        visualiser.draw_people([translated_track], blank_image, frame)

        return blank_image

    def chunk_to_video_scene(self, video, chunk, out_file, frames, label):
        capture = cv2.VideoCapture(video)

        mean = chunk[~np.all(chunk == 0, axis=2)].mean(axis=0)
        crop_size = int(min(500,  min(capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                                      capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        start_y = max(0, int(mean[0] - crop_size / 2))
        start_x = max(0, int(mean[1] - crop_size / 2))

        capture.set(cv2.CAP_PROP_POS_FRAMES, frames[0])

        track = self._chunk_to_track(chunk, frames)
        visualiser = TrackVisualiser()

        writer, frame_width, frame_height = self._create_writer(out_file)

        for i in range(len(chunk)):
            sucess, image = capture.read()
            cropped_image = self._draw_track_scene(
                track, frames[i], visualiser, image, start_x, start_y, crop_size, label)
            scaled_image = cv2.resize(cropped_image, (frame_width, frame_height))
            writer.write(scaled_image)

        writer.release()

    def _draw_track_scene(self, track, frame, visualiser,
                          original_image, start_x, start_y, crop_size, label):
        visualiser.draw_people([track], original_image, frame, offset_person=False)
        cropped_image = original_image[
            start_x:(start_x + crop_size), start_y:(start_y + crop_size)]
        visualiser.draw_frame_number(cropped_image, frame)
        visualiser.draw_text(cropped_image, label, position=(20, 450), color=(255, 255, 0))
        return cropped_image
