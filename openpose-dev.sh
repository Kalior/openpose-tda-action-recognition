docker run -it --rm \
    -v "$PWD":/openpose-action-recognition \
    -v "$PWD/media":/media \
    -v "$PWD/output":/output \
    -v "$PWD/datasets":/datasets \
    openpose-action-recognition \
    bash
