import gudhi as gd

import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Persistence(BaseEstimator, TransformerMixin):
    """Calculates persistence for the input point cloud using GUDHI.

    Parameters
    ----------
    max_edge_length : float, optional
        The max_edge_length passed to RipsComplex

    """

    def __init__(self, max_edge_length=0.5, use_rips=False):
        self.persistences = []
        self.max_edge_length = max_edge_length
        self.use_rips = use_rips

    def fit(self, X, y, **fit_params):
        """Returns self unchanged, as there are no parameters to fit.

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

    def transform(self, data):
        """Returns self unchanged, as there are no parameters to fit.

        Parameters
        ----------
        data : array-like
            shape = [n_points, n_keypoints, 3]

        Returns
        -------
        diags : array-like
            shape = [n_points, n_diags, 2]

        """
        return self.persistence(data)

    def visualise_point_clouds(self, data, number_of_points):
        """Uses pyplot to plot the point cloud used to calculate the persistence.

        Parameters
        ----------
        data : array-like, array of point-clouds.
        number_of_points : int, number of point-clouds to plot.

        """
        scaler = RobustScaler()
        scaler.fit(data.reshape(-1, 3))
        for i in range(number_of_points):
            points = scaler.transform(data[i])
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5)
            plt.show(block=False)

    def persistence(self, data):
        """Calcualtes persistence of the input data, point clouds.

        Parameters
        ----------
        data : array-like
            shape = [n_points, n_keypoints, 3]

        Returns
        -------
        diags : array-like
            shape = [n_points, n_diags, 2]

        """
        scaler = RobustScaler()
        scaler.fit(data.reshape(-1, 3))

        # self.visualise_point_clouds(data, 5)
        # plt.show()

        diags = np.zeros(data.shape[0], dtype=object)
        for i, d in enumerate(data):
            points = scaler.transform(d)

            if self.use_rips:
                diag = self._rips_complex(points)
            else:
                diag = self._alpha_complex(points)

            # Removing the points who don't die
            clean_diag = [p for p in diag if p[1][1] < np.inf]
            self.persistences.append(clean_diag)

            diags[i] = np.array([(p[1][1], p[1][0]) for p in clean_diag])

        return np.array(diags)

    def _rips_complex(self, points):
        rips = gd.RipsComplex(max_edge_length=self.max_edge_length, points=points)
        simplex_tree = rips.create_simplex_tree(max_dimension=3)
        diag = simplex_tree.persistence()
        return diag

    def _alpha_complex(self, points):
        alpha = gd.AlphaComplex(points=points)
        simplex_tree = alpha.create_simplex_tree(max_alpha_square=self.max_edge_length)
        diag = simplex_tree.persistence()
        return diag

    def _cubical_complex(self, points):
        shape = [points.shape[0]] * 2
        bitmap = np.zeros(shape)
        for i, p1 in enumerate(points):
            for j, p2 in enumerate(points):
                norm = np.linalg.norm(p1 - p2)
                bitmap[i, j] = norm

        cube = gd.CubicalComplex(top_dimensional_cells=bitmap.flatten(), dimensions=shape)
        diag = cube.persistence()
        return diag

    def _tangential_complex(self, points):
        tangential = gd.TangentialComplex(intrisic_dim=points.shape[1], points=points)
        simplex_tree = tangential.create_simplex_tree()
        diag = simplex_tree.persistence()
        return diag

    def save_persistences(self, out_dir):
        """Saves the persistence diagrams to file.

        Requires persistence to be called first.

        Parameters
        ----------
        out_dir : str, path to directory where the diagrams are saved.

        """
        for i, diag in enumerate(self.persistences):
            fig = gd.plot_persistence_diagram(diag)

            label = self.labels[i]
            plt.title(label)

            file_path = os.path.join(out_dir, 'persistence-{}.png'.format(i))
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()

    def save_betti_curves(self, out_dir):
        """Saves the betti curves of the calculate persistences to file.

        Requires persistence to be called first.

        Parameters
        ----------
        out_dir : str, path to directory where the curves are saved.

        """
        for i, diag in enumerate(self.persistences):
            tda_diag_df = self._construct_dataframe(diag)

            label = self.labels[i]
            plt.title(label)

            for dim in range(3):
                self._betti_curve(tda_diag_df, dim)

                file_path = os.path.join(out_dir, 'betti-curve-{}-{}.png'.format(i, dim))
                plt.savefig(file_path, bbox_inches='tight')
                plt.close()

    def _construct_dataframe(self, clean_diag_alpha):
        tda_diag_df = pd.DataFrame()

        tda_diag_df['Dimension'] = [el[0] for el in clean_diag_alpha]
        tda_diag_df['Birth'] = [el[1][0] for el in clean_diag_alpha]
        tda_diag_df['Death'] = [el[1][1] for el in clean_diag_alpha]
        tda_diag_df['Lifespan'] = tda_diag_df['Death'] - tda_diag_df['Birth']
        return tda_diag_df

    def _betti_curve(self, tda_diag_df, dim):
        betti_points = 100

        betti_curve_0 = []
        min_birth = tda_diag_df.loc[tda_diag_df.Dimension == dim].Birth.min()
        max_death = tda_diag_df.loc[tda_diag_df.Dimension == dim].Death.max()
        betti_range = np.linspace(min_birth, max_death, betti_points)
        for death in betti_range:
            nb_points_alive = tda_diag_df.loc[
                (tda_diag_df.Dimension == dim) & (tda_diag_df.Death >= death)].shape[0]
            betti_curve_0.append([death, nb_points_alive])
        betti_curve_0 = np.array(betti_curve_0)
        plt.plot(betti_curve_0[:, 0], betti_curve_0[:, 1])
