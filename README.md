# Action recognition based on OpenPose and TDA (using Gudhi and sklearn_tda)

## From video to action detection
The pipeline can be run in its entirety using the following scripts (also in `all.sh`):

```bash
python3.6 generate_tracks.py --video test-0.mp4 --output-directory output
python3.6 create_dataset.py --videos test-0.mp4 --tracks output/test-0-tracks.npz --out-file dataset/test
python3.6 visualise_dataset.py --dataset dataset/test --point-clouds
python3.6 train_classifier.py --dataset dataset/test --tda
```

The last step creates a trained classifier (in a `.pkl` file).  This classifier can then be used to generate predictions of actions live by running the script:

```bash
python3.6 live_prediction.py --classifier classifier.pkl --video test-0.mp4 --output-directory predictions
```

which will output the identified actions, a video with the predictions overlayed on the original video, and a video per predicted action.

Below is a description of what each of these steps does:

### generate_tracks.py
* Create a `tracker.Tracker` object with either `detector.CaffeOpenpose` (which is CMU's original implementation) or using `detector.TFOpenpose` (which is faster, but did not deliver the same level of accuracy for me). Also, requires an output directory to where it places the processed videos and tracks.
* Call the `video(path_to_video)` function on the `tracker.Tracker`.  The call will produce two files: A video file with the identified keypoints overlayed on the original video.  A file called `{path_to_video}-tracks.npz`, which contains two numpy arrays: `tracks` (i.e. the keypoints of each identified person in the video), and `frames` (i.e. the corresponding frame numbers for each identified person, primarily useful for later visualisation of the keypoints).

### create_dataset.py
* Run the `create_dataset.py` script to create a dataset from them.  If you have not previously labelled the data, the labelling process will either give you the option to look through the videos and discard bad chunks (if there are timestamps for the videos with corresponding labels) or manually label the data by displaying each chunk and requiring input on which label to attach to which chunk.  The script outputs `{name}-train.npz` and `{name}-test.npz` files containing the corresponding `chunks`, `frames`, `labels`, and `videos` of the train and test sets.  Note that the `frames` and `videos` are only used for visualisation of the data.  The labelling process only needs to be done once, after which a `.json` file is created per tracks file, which can be manually edited and will be parsed for labels subsequent times.
* If there are multiple datasets that you wish to combine, you can run the `combine_dataset.py` script which allows you to do exactly that.

### visualise_dataset.py
* If you wish, you can now run `visualise_dataset.py`, with any of the options to get an idea about, for instance, how the point clouds look or how well the features of the feature engineering seems to separate the different classes.

### train_classifier.py
* The final step is to run the `train_classifier.py` script.  It accepts a dataset as input (without the `-test` and `-train` suffix) and an option to run either `feature-engineering`, `tda`, or `ensemble` on the data.  They will produce confusion matrices of the classifier on the test set.  The `feature-engineering` option trains a classifier on hand-selected features.  The `tda` option runs a Sliced Wasserstein Kernel on the Persistence diagrams of the generated point clouds from the data.  The `ensemble` option combines the Sliced Wasserstein kernel with other features for a better classification.

### live_prediction.py
* Given a trained classifier, this script uses the `tracker.Tracker` to yield identified tracks from the tracking of people in the video.  On each such track, it does post-processing (using `analysis.PostProcessor`) step and then divides the track into chunks and predicts actions.  If the classifier is not sufficiently confident in a classification, the prediction is discarded.

## Dockerfiles
There are currently four dockerfiles, corresponding to three natural divisions of dependencies, and one with every dependency:

* `dockerfiles/Dockerfile-openpose-gpu`: which is the GPU version of OpenPose, allows the openpose parts of this project to be run.
* `dockerfiles/Dockerfile-openpose-cpu`: which is the CPU version of OpenPose.
* `dockerfiles/Dockerfile-tda`: which contains the `Gudhi` and `sklearn_tda` for the classification part of the project.
* `Dockerfile`: which installs both openpose (assuming a GPU) as well as the TDA libraries.  This file can do with some cleanup using build stages.

After building the Dockerfiles, there is a script `dev.sh` which runs the container and mounts the source directory as well as the expected locations of the data.  It is provided more out of convenience than anything else and may need some modification depending on your configuration.

## Recording videos
There is a helper script for producing timestamps for labels while recording videos.  It is called `record_videos.py` and requries a video name, a path to the camera device and a video size.  It prompts the user in multiple steps: First, asks whether to record video or stop recording.  Second, it prompts the user for a label for the timestamp.  These steps are repeated until a quit command is given.
