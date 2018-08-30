"""
Action recognition using OpenPose and TDA
=========================================
The TDA tools used are Gudhi and sklearn_tda.

The pipeline, in short, is: estimate poses using OpenPose, label the data,
transform into point clouds, calculate persistence using Gudhi and
finally use kernels and vectorisations from sklearn_tda to predict
actions.
"""

__all__ = ['analysis', 'augmentors', 'classifiers',
           'detector', 'features', 'tracker', 'transforms', 'util']
