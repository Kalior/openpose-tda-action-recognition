# Action recognition based on [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose/) and TDA (using [Gudhi](http://gudhi.gforge.inria.fr/python/latest/) and [sklearn_tda](https://github.com/MathieuCarriere/sklearn_tda))

## Performance

The ensemble classifier achieves an accuracy of `0.912` on custom data.  However, there is still a need to capture more data to see how well it would generalise over different actors and scenes.  The TDA classifier on its own achieves an accuracy of `0.823` on the same data.

## From video to action detection

The pipeline can be run in its entirety using the following scripts (also in [`all.sh`](all.sh), and see `--help` for each script for options and parameters).  The first step is to generate data and train a classifier:

```bash
python3.6 generate_tracks.py --video test-0.mp4 --out-directory output
python3.6 create_dataset.py --videos test-0.mp4 --tracks output/test-0-tracks.npz
python3.6 visualise_dataset.py --dataset dataset/dataset --point-clouds
python3.6 train_classifier.py --dataset dataset/dataset --tda
```

The last step creates a trained classifier (in a `.pkl` file).  This classifier can then be used to generate predictions of actions live by running the script:

```bash
python3.6 live_prediction.py --classifier classifier.pkl --video test-0.mp4
```

The script will output identified actions, a video with the predictions overlayed on the original video, and a video per predicted action.

Most of what the scripts do is to wrap input and output to the different modules in [`action_recognition`](action_recognition/).  The final script (`live_prediction.py`) being the exception, as it also aggregates some of the predictions, and decides on which parts of each track should be used to predict actions.  Below is a description of what each of these scripts do:

#### generate_tracks.py

* The [`generate_tracks.py`](generate_tracks.py) script creates tracks of people identified in a video for dataset creation.
* Creates a [`tracker.Tracker`](action_recognition/tracker/tracker.py) object with either [`detector.CaffeOpenpose`](action_recognition/detector/caffe_openpose.py) (which is CMU's original implementation) or using [`detector.TFOpenpose`](action_recognition/detector/tf_openpose.py) (which is faster, but did not deliver the same level of accuracy for me). Also, requires an output directory to where it places the processed videos and tracks.
* Produces two files: A video file with the identified keypoints overlayed on the original video.  A file called `{path_to_video}-tracks.npz`, which contains two numpy arrays: `tracks` (i.e. the keypoints of each identified person in the video), and `frames` (i.e. the corresponding frame numbers for each identified person, primarily useful for later visualisation of the keypoints).
* Each track is a `[n_frames, n_keypoints, 3]` `numpy.ndarray` which is predicted as being a single person through several frames, making the final outputted array of shape `[n_tracks, n_frames, n_keypoints, 3]`, where the values are (x, y, confidence). The frames array, correspondingly, has the shape `[n_tracks, n_frames, 1]`. Note, however, that both a arrays will be ndarrays with `dtype=object`, since `n_frames` per track will differ.

#### create_dataset.py

* Run the [`create_dataset.py`](create_dataset.py) script to create a dataset from tracks.  If you have not previously labelled the data, the labelling process will either give you the option to look through the videos and discard bad chunks (if there are timestamps for the videos with corresponding labels) or manually label the data by displaying each chunk and requiring input on which label to attach to which chunk.
* The script outputs `{name}-train.npz` and `{name}-test.npz` files containing the corresponding `chunks`, `frames`, `labels`, and `videos` of the train and test sets.  Note that the `frames` and `videos` are only used for visualisation of the data.
* The labelling process only needs to be done once, after which a `.json` file is created per tracks file, which can be manually edited and will be parsed for labels subsequent times.
* During the creation of the dataset, before dividing tracks into chunks, there are also a couple of post-processing steps:
        1. Merge tracks that are very close to each other at their ends, or throughout a longer period.
        2. Remove tracks that are too short, under 15 frames long.
        3. Fill-in missing keypoints (as OpenPose sometimes does not output every keypoint) by maintaining the same distance to a connected keypoint (e.g. wirst-elbow) as when the keypoint was last seen.  This increases the accuracy of the classifier later on.
        4. Fill-in missing frames by interpolating positions of every keypoint. This makes sure that chunks don't incorrectly have more movement just because OpenPose lost track of that person for a couple of frames.
* If there are multiple datasets that you wish to combine, you can run the [`combine_datasets.py`](combine_datasets.py) script which allows you to do exactly that.

#### visualise_dataset.py

* If you wish, you can now run [`visualise_dataset.py`](visualise_dataset.py) to get an idea about, for instance, how the point clouds look or how well the features of the feature engineering seems to separate the different classes.

#### train_classifier.py

* The [`train_classifier.py`](train_classifier.py) script finally train a classifier on the data.  It accepts a dataset as input (without the `-test` and `-train` suffix) and an option to run either [`--feature-engineering`](action_recognition/classifiers/feature_engineering_classifier.py), [`--tda`](action_recognition/classifiers/tda_classifier.py), or [`--ensemble`](action_recognition/classifiers/ensemble_classifier.py).  They will produce confusion matrices of the classifier on the test set.  The `--feature-engineering` option trains a classifier on hand-selected features.  The `tda` option runs a SlicedWasserstein Kernel on the Persistence diagrams of the generated point clouds from the data.  The `ensemble` option combines the Sliced Wasserstein kernel with the feature engineering using a voting classifier.
* The pipeline for the TDA calculation has 7 steps, remember the data is split up into chunks by [`create_dataset.py`](create_dataset.py):
    1. Extract certain keypoints (the neck, ankles, and wrists have worked the best for me), which both speeds up the computation and increases accuracy.
    2. Smooth the path of each keypoint.  This is mainly done since OpenPose sometimes produces jittery output, and this helps to remove that (and increases accuracy as a result).
    3. Normalise every chunk so that it is centered around `(0, 0)`.
    4. Flatten the chunks from shape `[n_frames, n_keypoints, 2]` to `[n_keypoints * n_frames, 3]`.  The third dimension corresponds to the index of the frame, not actual time.
    5. Calculate persistence using `Gudhi`'s `AlphaComplex` (with `max_alpha_square` set to 2).
    6. Calculate the `SlicedWasserstein` kernel from `sklearn_tda`.
    7. Train a `scikit-learn` `SVC` classifier.
* The training can take a couple of minutes, naturally longer for the TDA calculations than for the pure feature engineering.  The SlicedWasserstein kernel is the computation that takes the longest (but thankfully prints its progress), roughly 1.6 times longer than the next most time-consuming operation, which is the persistence calculation of the AlphaComplex which takes place just before.

#### live_prediction.py
* [`live_prediction.py`](live_prediction.py) takes a trained classifier and uses [`tracker.Tracker`](action_recognition/tracker/tracker.py) to yield identified tracks from the tracking of people in the video.
* On each such track (every 20:th frame), it does post-processing (using [`analysis.PostProcessor`](action_recognition/analysis/post_processor.py)) and then takes the latest 50, 30, 25, and 20 (arbitrarily) frames as chunks for which actions are predicted.  The most likely action (highest probability/confidence from the classifier) from all chunks is selected as the action for the person.
* If the confiedence for a classification falls below a user-specifed threshold, the prediction is discarded.
* It also tries to predict if a person moves through e.g. a checkout-area without stopping by identifying if a person moves during several consecutive frames.

## Dockerfiles

There are currently four dockerfiles, corresponding to three natural divisions of dependencies, and one with every dependency:

* [`dockerfiles/Dockerfile-openpose-gpu`](dockerfiles/Dockerfile-openpose-gpu): which is the GPU version of OpenPose, allows the openpose parts of this project to be run.
* [`dockerfiles/Dockerfile-openpose-cpu`](dockerfiles/Dockerfile-openpose-cpu): which is the CPU version of OpenPose.
* [`dockerfiles/Dockerfile-tda`](dockerfiles/Dockerfile-tda): which contains `Gudhi` and `sklearn_tda` for the classification part of the project.
* [`Dockerfile`](Dockerfile): which installs both openpose (assuming a GPU) as well as the TDA libraries.  This file can do with some cleanup using build stages.

After building the Dockerfiles, there is a script [`dev.sh`](dev.sh) which runs the container and mounts the source directory as well as the expected locations of the data.  It is provided more out of convenience than anything else and may need some modification depending on your configuration.

## Recording videos

There is a helper script for producing timestamps for labels while recording videos.  It is called [`record_videos.py`](record_videos.py) and requires a video name, a path to the camera device and video size.  It prompts the user in multiple steps: First, asks whether to record video or stop recording.  Second, it prompts the user for a label for the timestamp.  These steps repeat until the user quits the program.  The produced timestamps are read by [`create_dataset.py`](create_dataset.py) to help reduce labelling time.

## Issues with the approach

* A bit slow - OpenPose takes 0.5 s/frame, and the TDA classifier takes 3 s/person and prediction.  This time complexity comes mainly from the kernel calculation from sklearn_tda, and the persistence calculation of the gudhi library.  Both of these have parameters that can be tuned (see [TDAClassifier.\_pre_validated_pipeline()](action_recognition/classifiers/tda_classifier.py#L113)), at the expense of accuracy.
* We are restricted to 2D positions - a limitation from OpenPose, which makes classification harder.
* OpenPose can be quite jittery, especially when using lower resolutions.
* TDA does not have any way of recognising still vs lying.  Since the actions don't have any movement, they don't form any shapes that TDA can recognise.
* TDA also does not have a concept of direction, only the shape of the point cloud. Therefore, a vertical action can easily be confused with a horizontal one.
* While the final confusion matrix/accuracy looks good, I am worried that the data/actions are too easy since the feature engineering works so well.  The TDA kernel might generalise better?
