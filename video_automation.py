import ffmpy
from threading import Thread
from time import sleep, time
import argparse
import json
import os
import subprocess


def main(args):
    record = True
    i = 0
    while record:
        val = input("(n)ew video, (d)one\n")
        if val == 'n':
            out_file_video = "{}-{}.mp4".format(args.video_name, i)
            out_file_timestamps = "{}-{}-timestamps.json".format(args.video_name, i)
            ff, start_time = start_recording(out_file_video, args.video_path, args.frame_rate)
            # Sleep in order to allow the program some time to start the recording.
            sleep(2)

            timestamps = label_recording(start_time, args.frame_rate)

            ff.process.terminate()
            i += 1
            sleep(2)
            with open(out_file_timestamps, 'w') as f:
                json.dump(timestamps, f)

        elif val == 'd':
            record = False


def label_recording(video_start_time, frame_rate):
    timestamps = []
    labelling = True
    while labelling:
        keypress = input('Label? (Scan, Cash, sTill, Moving, Lie, Other, quit)')
        if keypress in ['s', 'c', 'o', 'm', 't', 'l']:
            start_time = time() - video_start_time
            label = parse_keypress_to_label(keypress)

            input('Press any key to stop')
            end_time = time() - video_start_time
            timestamp = {
                'start_time': start_time,
                'end_time': end_time,
                'start_frame': int(start_time * frame_rate),
                'end_frame': int(end_time * frame_rate),
                'label': label
            }
            timestamps.append(timestamp)
        elif keypress == 'q':
            labelling = False
    return timestamps


def parse_keypress_to_label(keypress):
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
    else:
        label = 'other'
    return label


def start_recording(out_file, video_path, frame_rate):
    ff = ffmpy.FFmpeg(
        global_options=['-video_size 1280x720', '-r {}'.format(frame_rate), '-loglevel panic'],
        inputs={video_path: None},
        outputs={out_file: None}
    )

    def _execute():
        try:
            ff.run()
        except ffmpy.FFRuntimeError as ex:
            # In case of normal exit, we terminated the process
            if ex.exit_code and ex.exit_code != 255:
                raise

    process = Thread(target=_execute)
    process.start()
    start_time = time()

    return ff, start_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Helper script for labelling data in record(ing)-time.')
    parser.add_argument('--video-name', type=str, help='Prefix of the outputted video files.')
    parser.add_argument('--video-path', type=str, default='/dev/video1',
                        help='Path to the video device (e.g. /dev/video1)')
    parser.add_argument('--frame-rate', type=int, default=30,
                        help='Sets the frame rate of the recording.')

    args = parser.parse_args()

    main(args)
