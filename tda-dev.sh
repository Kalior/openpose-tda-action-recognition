docker run -it --rm \
    -v "$PWD":/tda-action-recognition \
    -v "$PWD/media":/media \
    -v "$PWD/output":/output \
    -v "$PWD/datasets":/datasets \
    tda-action-recognition \
    bash
