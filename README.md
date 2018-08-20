# Action recognition based on OpenPose and TDA (using Gudhi and sklearn_tda)

## From video to action detection
The pipeline can be run in its entirety using the following four scripts (also in `all.sh`):

```

python3.6 generate_tracks.py --video test-0.mp4 --output-directory output
python3.6 create_dataset.py --videos test-0.mp4 --tracks output/test-0-tracks.npz --out-file dataset/test
python3.6 run_tda.py --dataset dataset/test --visualise
python3.6 run_tda.py --dataset dataset/test --tda


```

Below is a description of what each of these steps do:

* Create a `tracker.Tracker` object with either `detector.CaffeOpenpose` (which is CMU's original implementation) or using `detector.TFOpenpose` (which is faster, but did not deliver the same level of accuracy for me). Also specify an output directory for where the processed video and tracks will be created.
* Call the `video(path_to_video)` function on the `tracker.Tracker`.  This will produce two files: A video file with the identified keypoints overlayed on the original video. A file called `{path_to_video}-tracks.npz`, which contains two numpy arrays: `tracks` (i.e. the keypoints of each identified person in the video), and `frames` (i.e. the corresponding frame numbers for each identified person, primarily useful for later visualisation of the keypoints).
* These first two steps have two correspodning scripts: `generate_tracks.py` which will generate the tracks for a single video, and `automation.py` which (at the moment) requires some manual editing, but generates tracks for every video in a folder.
* Run the `create_dataset.py` script on the videos and tracks you which to create a dataset out of.  If you have not previously labelled the data.  The labelling process will either give you the option to look through the videos and discard bad chunks (if there are timestamps for the videos with corresponding labels), or manually label the data by displaying each chunk and requiring input on which label to attach to which chunk.  The output of the script is a `{name}-train.npz` and a `{name}-test.npz` files containing the corresponding `chunks`, `frames`, `labels`, and `videos`.  Note that the `frames` and `videos` are only needed for visualisation of the data.  The labelling process only needs to be done once, after which a `.json` file is created per tracks file, which can be manually edited and will be read for labels subsequent times.
* If there are multiple datasets that you wish to combine, you can run the `combine_dataset.py` script which allows you to do exactly that.
* The final step is to run the `run_tda.py` script.  It accepts a dataset as input (without the `-test` or `-train` suffix) and an option to either run either `visualise`, `mapper`, `tda`, or `ensemble` on the data.  The `tda` and `ensemble` option will produce confusion matricies of the test set.  The `tda` option runs a Sliced Wasserstein Kernel on the Persistence diagrams of the generated point clouds from the data.  The `ensemble` option combines the Sliced Wasserstein kernel with other features for a better classification.  The `mapper` option runs the Mapper algorithm on the combined dataset.  The `visualise` option displays the average keypoint position for each class and the features used in the `ensemble` for each class.

## Dockerfiles
There are currently four dockerfiles, as there are three natural divisions of the project:

* `dockerfiles/Dockerfile-openpose-gpu`: which is the GPU version of OpenPose, which allows the openpose parts of this project to be run.
* `dockerfiles/Dockerfile-openpose-cpu`: which is the CPU version of OpenPose.
* `dockerfiles/Dockerfile-tda`: which contains the `Gudhi` and `sklearn_tda` for the classification part of the project.
* `Dockerfile`: which installs both openpose (assuming a GPU) as well as the TDA libraries.  This file can really do with some cleanup using build stages.

After building the Dockerfiles there is a script `dev.sh` which runs the corresponding container and mounts the source directory as well as the expected locations of the data.  It is provided more out of convenience than anything, and may need some modification depending on your configuration.
