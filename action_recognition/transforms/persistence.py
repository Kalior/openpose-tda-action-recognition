import gudhi as gd

import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Persistence(BaseEstimator, TransformerMixin):

    def __init__(self, max_edge_length=0.5):
        self.persistences = []
        self.max_edge_length = max_edge_length

    def fit(self, X, y, **fit_params):
        return self

    def transform(self, data):
        return self.persistence(data)

    def visualise_point_clouds(self, data, number_of_points):
        scaler = RobustScaler()
        scaler.fit(data.reshape(-1, 3))
        for i in range(number_of_points):
            points = scaler.transform(data[i])
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5)
            plt.show(block=False)

    def persistence(self, data):
        dim = 3
        betti_numbers = np.zeros((data.shape[0], dim))
        scaler = RobustScaler()
        scaler.fit(data.reshape(-1, 3))

        diags = []
        for i, d in enumerate(data):
            points = scaler.transform(d)

            rips = gd.RipsComplex(max_edge_length=self.max_edge_length, points=points)
            simplex_tree = rips.create_simplex_tree(max_dimension=3)

            # alpha = gd.AlphaComplex(points=points)
            # simplex_tree = alpha.create_simplex_tree(max_alpha_square=0.1)

            diag_alpha = simplex_tree.persistence()
            # Removing the points who don't die
            clean_diag_alpha = [p for p in diag_alpha if p[1][1] < np.inf]
            self.persistences.append(clean_diag_alpha)
            # tda_diag_df = self._construct_dataframe(clean_diag_alpha)
            # self._betti_curve(tda_diag_df)

            diags.append(np.array([(p[1][1], p[1][0]) for p in clean_diag_alpha]))
            betti = simplex_tree.betti_numbers()
            # Make sure we fill the 2 dimensions
            pad = np.pad(betti, (0, dim - len(betti)), 'constant')
            betti_numbers[i] = pad

        return np.array(diags)

    def save_persistences(self, out_dir):
        for i, diag in enumerate(self.persistences):
            fig = gd.plot_persistence_diagram(diag)

            label = self.labels[i]
            plt.title(label)

            file_path = os.path.join(out_dir, 'persistence-{}.png'.format(i))
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()

    def save_betti_curves(self, out_dir):
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
