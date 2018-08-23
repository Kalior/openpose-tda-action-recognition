import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ConcatenatePointClouds(BaseEstimator, TransformerMixin):
    """Concatenates two point clouds together (as np.hstack does not seem to work on object arrays).

    Parameters
    ----------
    number_of_clouds : int
        The number of point clouds that are to be concatenated.
    """

    def __init__(self, number_of_clouds):
        self.number_of_clouds = number_of_clouds

    def fit(self, X, y=None, **fit_params):
        """Returns self, as there are no parameters to fit.

        Parameters
        ----------
        X : ignored
        y : ignored
        fit_params : ignored

        Returns
        -------
        self : unchanged

        """
        return self

    def transform(self, clouds):
        """Transforms the clouds into a single point cloud with increased dimensionality.

        Each cloud has to be stackable with the corresponding clouds in the other
        parts of the data.

        Parameters
        ----------
        clouds : array-like
            The clouds to combine.
            expected shape = [n_clouds * n_points, n_keypoints, x],
            but probably [n_clouds * n_points, ] with dtype=object since
            we don't enforce a specific number of frames per cloud.

        Returns
        -------
        clouds : array-like
            The clouds concatenated.

        """
        print("Concatenate point clouds")
        individual_clouds = np.split(clouds, self.number_of_clouds)
        return np.array([np.hstack(cs) for cs in zip(*individual_clouds)])
