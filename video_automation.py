import ffmpy
from threading import Thread
from time import sleep

file_name = "cash-"

run = True
i = 0
while run:
    val = input("(n)ew, (d)one\n")
    if val == 'n':
        out_file = file_name + str(i) + ".mp4"
        ff = ffmpy.FFmpeg(
            global_options=['-video_size 640x480'],
            inputs={'/dev/video1': None},
            outputs={out_file: None}
        )

        def _execute():
            try:
                ff.run()
            except ffmpy.FFRuntimeError as ex:
                if ex.exit_code and ex.exit_code != 255:
                    raise
        process = Thread(target=_execute)
        process.start()

        input('press enter to stop\n')
        ff.process.terminate()
        i += 1
        sleep(2)
    elif val == 'd':
        run = False
