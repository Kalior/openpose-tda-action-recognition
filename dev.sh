docker run -it --rm \
    -v "$PWD":/openpose-action-recognition \
    -v "$PWD/../media":/media \
    -v "$PWD/../output":/output \
    -v "$PWD/../dataset":/dataset \
    openpose-action-recognition \
    bash
