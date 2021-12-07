from abc import ABC
import time
import warnings
from itertools import product, repeat
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as AA
import matplotlib.colors as cls
from mpl_toolkits.axes_grid1 import make_axes_locatable

import statsmodels.api as smapi

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from sklearn.mixture import GaussianMixture


from src.Utils import validate_dir, frequency_bins
from src.DataLoader import transform_path
from src.Kernel import *
from src.Model import RegionModel, JMAK


class ParameterMap(ABC):
    class KernelWrapper:
        def __init__(self, func, params):
            self.func = func
            self.params = params

        def __call__(self, *args, **kwargs):
            return self.func(*args, *self.params)

    def __init__(
            self,
            rmodel,
            bio_data,
            discretise_bio=True,
            randomise=False,
            num_bins=50,
            val_frac=.2,
            rm_percentile=5.,
            random_state=None,
            verbosity=1
    ):
        if len(rmodel.models) == 1:
            raise ValueError('The region model must contain at least two JMAK models.')

        if len(rmodel.models) != len(bio_data):
            raise ValueError('The number of region models must be equal to the number of biological data points.')
        self.rmodel = rmodel
        self.learnt_var = 0
        self.discretise_bio = discretise_bio
        self.val_frac = val_frac
        self.random_state = random_state
        m_shape = np.asarray(list(self.rmodel.get_model_parameter('m')))
        lower, upper = np.percentile(m_shape, [rm_percentile / 2., 100. - rm_percentile / 2.])
        m_shape_mask = np.logical_and(m_shape > lower, m_shape < upper)

        max_frac = np.asarray(list(self.rmodel.get_model_parameter('max_frac')))
        lower, upper = np.percentile(max_frac, [rm_percentile / 2., 100. - rm_percentile / 2.])
        max_frac_mask = np.logical_and(max_frac > lower, max_frac < upper)

        beta = np.asarray(list(self.rmodel.get_model_parameter('beta')))
        lower, upper = np.percentile(beta, [rm_percentile / 2., 100. - rm_percentile / 2.])
        beta_mask = np.logical_and(beta > lower, beta < upper)

        param_mask = np.logical_and(m_shape_mask, np.logical_and(max_frac_mask, beta_mask))
        self.all_bio_data = bio_data[param_mask]

        m_shape, self.m_mean, self.m_std = ParameterMap.normalise(m_shape[param_mask])
        max_frac, self.max_frac_mean, self.max_frac_std = ParameterMap.normalise(max_frac[param_mask])
        beta, self.beta_mean, self.beta_std = ParameterMap.normalise(beta[param_mask])

        self.all_data_params = np.asarray([m_shape, max_frac, beta]).T

        if randomise:
            np.random.shuffle(self.all_bio_data)

        if self.discretise_bio:
            self.bio_bins = frequency_bins(self.all_bio_data, num_bins)
            # Increase border of last bin to include last data point
            self.bio_bins[-1] += .1
            self.all_bio_data = self.discretise(self.all_bio_data)

        self.data_params, self.data_params_val, self.bio_data, self.bio_data_val = None, None, None, None
        self.reshuffle()
        self.pca = PCA(n_components=2)
        self.param_pca = self.pca.fit(self.data_params).transform(self.data_params).T
        if verbosity > 0:
            print(
                'Explained parameter ratio:\tPC1 %.2f\tPC2: %.2f' %
                (self.pca.explained_variance_ratio_[0],
                 self.pca.explained_variance_ratio_[1])
            )

    def reshuffle(self):
        self.data_params, self.data_params_val, self.bio_data, self.bio_data_val = train_test_split(
            self.all_data_params,
            self.all_bio_data,
            test_size=self.val_frac,
            random_state=self.random_state
        )

    @staticmethod
    def normalise(data):
        std = np.std(data)
        if std == 0:
            mean = np.min(data)
            std = np.max(data)
        else:
            mean = np.mean(data)
        data -= mean
        data /= std
        return data, mean, std

    def convert(self, params_):
        params = np.copy(params_)
        params[:, 0] = (params[:, 0] - self.m_mean) / self.m_std
        params[:, 1] = (params[:, 1] - self.max_frac_mean) / self.max_frac_std
        params[:, 2] = (params[:, 2] - self.beta_mean) / self.beta_std
        return params

    @staticmethod
    def error(real_data, est_mean, is_regression=True):
        if is_regression:
            s = (np.asarray(real_data) - np.asarray(est_mean))**2
        else:
            s = real_data != est_mean
        return np.mean(s)

    def plot_parameter_map(
            self,
            plotted_dp=100,
            levels=15,
            figsize=(8, 7),
            verbosity=0,
            save_fig=True,
            save_prefix=''
    ):
        params_val_pca = self.pca.transform(self.data_params_val).T
        plt.figure(figsize=figsize)
        map_contour = plt.tricontourf(*self.param_pca, self.bio_data, levels, cmap='seismic', alpha=.7)
        plt.tricontour(*self.param_pca, self.bio_data, levels, linewidths=0.5, colors='k')

        size = np.minimum(plotted_dp, len(self.bio_data_val))
        idx = np.random.choice(len(self.bio_data_val), size=size, replace=False)
        plt.scatter(
            *params_val_pca.T[idx].T,
            c=self.bio_data_val[idx],
            cmap='seismic',
            edgecolor='white',
            norm=cls.Normalize(vmin=np.min(self.bio_data), vmax=np.max(self.bio_data))
        )

        plt.title('Learnt parameter map: %s' % self.rmodel.name)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        cbar = plt.colorbar(map_contour)
        cbar.set_alpha(1)
        cbar.draw_all()

        if save_fig:
            directory = validate_dir('figures/predict_models')
            plt.savefig('%s/%s_%s_learnt_map.png' % (directory, save_prefix, self.rmodel.name))
            plt.close('all')
        else:
            plt.show()

    def plot_error(
            self,
            mean,
            true_val,
            var,
            params,
            convert_params=True,
            convert_val=True,
            save_fig=True,
            save_prefix=''
    ):
        if convert_params:
            params = self.convert(params)
        if convert_val:
            true_val = self.discretise(true_val)
        pca_params = self.pca.transform(params)
        fig = plt.figure(figsize=(8, 7))
        ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=2)
        if var is not None:
            var = np.maximum(var, 6**2)
        cbar = ax1.scatter(
            *pca_params.T,
            c=mean - true_val,
            s=var if var is not None else 6**2,
            cmap='seismic',
            edgecolor='black'
        )
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(cbar, cax=cax, orientation='vertical')
        ax1.set_title('Median absolut prediction error: %.1f' % np.median(mean - true_val))
        ax1.set_xlabel('PC 1')
        ax1.set_ylabel('PC 1')
        ax2 = plt.subplot2grid((3, 4), (0, 3), rowspan=2)
        ax2.hist(
            mean - true_val,
            bins=self.bio_bins.size if self.discretise_bio else 'auto',
            range=cbar.get_clim(),
            orientation='horizontal'
        )
        ax2.set_title('Deviance')
        ax2.set_xlabel('#data')
        ax2.set_yticks([])
        ax2.set_yticklabels([])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.set_ylim(cbar.get_clim())
        ax3 = plt.subplot2grid((3, 4), (2, 0), colspan=4)
        error = self.error(real_data=true_val, est_mean=mean, is_regression=not self.discretise_bio)
        num_bins = np.minimum(20, self.bio_bins.size) if self.discretise_bio else 20
        error_name = 'Absolute error' if self.discretise_bio else 'MSE'
        error_dist = np.abs(true_val - mean) if self.discretise_bio else (true_val - mean)**2
        ax3.hist(error_dist, bins=num_bins)
        ax3.set_title('%s: %.3f' % (error_name, error))
        ax3.set_xlabel(error_name)
        ax3.set_ylabel('#data')
        fig.tight_layout()
        if save_fig:
            directory = validate_dir('figures/predict_models')
            plt.savefig('%s/%s_%s_error_distribution.png' % (directory, save_prefix, self.rmodel.name))
            plt.close('all')
        else:
            plt.show()

    def discretise(self, bio_data):
        return np.maximum(np.minimum(np.digitize(bio_data, self.bio_bins) - 1, bio_data - 2), 0)

    def learn(self, verbosity=0):
        pass

    def estimate(self, new_params, convert_dp=True):
        pass


# ######################################
# Non-parametric approaches
# ######################################
class GPParameterMap(ParameterMap):
    def __init__(
            self,
            rmodel,
            bio_data,
            randomise=False,
            num_bins=10,
            val_frac=.2,
            rm_percentile=5.,
            random_state=None,
    ):
        super().__init__(
            rmodel,
            bio_data,
            discretise_bio=False,
            randomise=randomise,
            num_bins=num_bins,
            val_frac=val_frac,
            rm_percentile=rm_percentile,
            random_state=random_state
        )
        self.gpr = GaussianProcessRegressor(kernel=RBF() + WhiteKernel(), random_state=random_state)

    def learn(self, verbosity=0):
        self.gpr.fit(self.data_params, self.bio_data)
        if verbosity > 0:
            print('R2: %.3f' % self.gpr.score(self.data_params, self.bio_data))

    def estimate(self, new_params, convert_dp=True):
        if convert_dp:
            new_params = self.convert(new_params)
        mean, std = self.gpr.predict(new_params, return_std=True)
        return mean, std**2


class KNNParameterMap(ParameterMap):
    def __init__(
            self,
            rmodel,
            bio_data,
            k_neighbours=5,
            randomise=False,
            num_cpus=1,
            num_bins=10,
            val_frac=.2,
            rm_percentile=5.,
            random_state=None,
    ):
        super().__init__(
            rmodel,
            bio_data,
            discretise_bio=True,
            randomise=randomise,
            num_bins=num_bins,
            val_frac=val_frac,
            rm_percentile=rm_percentile,
            random_state=random_state
        )
        self.k_neighbours = k_neighbours
        self.num_cpus = num_cpus
        self.knn = KNeighborsClassifier(n_neighbors=k_neighbours, n_jobs=self.num_cpus)

    def learn(self, verbosity=0):
        self.knn.fit(self.data_params, self.bio_data)

    def estimate(self, new_params, convert_dp=True):
        if convert_dp:
            new_params = self.convert(new_params)
        distr = self.knn.predict_proba(new_params)
        mean = np.argmax(distr, axis=1)
        values = np.ones_like(distr) * np.arange(distr.shape[1])
        var = np.sum((values - mean.reshape(-1, 1))**2 * distr, axis=1)
        return mean, var


# ######################################
# Parametric approaches
# ######################################
class LinearParameterMap(ParameterMap):
    def __init__(
            self,
            rmodel,
            bio_data,
            randomise=False,
            num_bins=10,
            val_frac=.2,
            rm_percentile=5.,
            random_state=None,
    ):
        super().__init__(
            rmodel,
            bio_data,
            discretise_bio=True,
            randomise=randomise,
            num_bins=num_bins,
            val_frac=val_frac,
            rm_percentile=rm_percentile,
            random_state=random_state
        )
        self.lin_est = None
        self.lin_params = None

    def learn(self, verbosity=0):
        self.lin_est = smapi.OLS(
            self.bio_data,
            smapi.add_constant(self.data_params)
        )
        result = self.lin_est.fit()
        self.lin_params = result.params
        if verbosity > 0:
            print('Linear regression parameters: %s' % result.params)
            print('MSE: %.2f' % result.mse_total)
            print('Adjusted R2: %.2f' % result.rsquared_adj)

    def estimate(self, new_params, convert_dp=True):
        if convert_dp:
            new_params = self.convert(new_params)

        mean = np.round(
            self.lin_est.predict(
                self.lin_params,
                exog=smapi.add_constant(new_params)
            )).astype('int')

        return mean, None
