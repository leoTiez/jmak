import numpy as np
import time
import warnings
import pandas as pd
from itertools import product, repeat
import statsmodels.api as smapi
import matplotlib.pyplot as plt
import matplotlib.colors as cls
from mpl_toolkits.axes_grid1 import make_axes_locatable, host_subplot
import mpl_toolkits.axisartist as AA
from abc import ABC
import multiprocessing
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA

import keras
from tensorflow.keras import layers
import tensorflow as tf

from src.Utils import validate_dir, frequency_bins
from src.DataLoader import transform_path
from src.Kernel import *


class JMAK:
    def __init__(
            self,
            time_points,
            data_points,
            m=None,
            max_frac=None,
            beta=None,
            name='',
            min_m=.01,
            handle_stationary=False
    ):
        self.time_points = np.asarray(time_points)
        self.data_points = np.asarray(data_points)
        if handle_stationary:
            if np.all(self.data_points[-1] == self.data_points[-2]):
                self.data_points = self.data_points[:-1]
                self.time_points = self.time_points[:-1]
        self.time_points = self.time_points.reshape(-1)
        self.data_points = self.data_points.reshape(-1)
        self.m = m
        self.max_frac = max_frac
        self.beta = beta
        self.name = name
        self.min_m = min_m

    def repair_fraction(self, time):
        if self.m is None or self.max_frac is None or self.beta is None:
            raise ValueError(
                'Model parameters have not been set.\n'
                'Run estimate_parameters to fit the model to the data, or set the parameters manually'
            )

        return (1 - np.exp(- np.power(self.beta * time, self.m))) * self.max_frac

    def repair_fraction_over_time(self, to_time):
        return np.asarray([self.repair_fraction(t) for t in np.arange(to_time)])

    def _estimate_shape_scale(self, max_frac):
        if np.any(max_frac < self.data_points):
            return np.nan, np.nan, np.nan
        dp = self.data_points.copy()
        dp[dp == max_frac] -= np.finfo('float').eps
        dp[dp == 0] += np.finfo('float').eps
        lin_est = smapi.OLS(
            np.log(np.log(1. / (1 - dp / max_frac))),
            smapi.add_constant(np.log(self.time_points))
        )
        result = lin_est.fit()

        if result.params[1] > self.min_m:
            return (
                result.params[1],  # shape
                np.exp(result.params[0] / result.params[1]),  # scale
                result.fvalue  # fvalue
            )
        else:
            return np.nan, np.nan, np.nan

    def estimate_parameters(
            self,
            min_f,
            max_f=1.,
            delta_f=.01,
            fval_accuracy=1e-3,
            verbosity=0,
            figsize=(8, 7),
            save_fig=True,
            save_prefix=''
    ):
        if max_f > 1 or max_f < 0:
            raise ValueError('Maximum fraction cannot be larger than 1 or lower than 0')
        if delta_f > max_f - min_f or delta_f < 0:
            raise ValueError('Fraction increase (delta f) must be lower than the difference between min_f and max_f and'
                             'greater than 0')

        fval = []
        beta_est = []
        m_est = []
        if verbosity > 1:
            num_cols = np.minimum(int(np.sqrt((max_f - min_f) / delta_f)), 3)
            num_rows = np.minimum(int(np.ceil(((max_f - min_f) / delta_f) / num_cols)), 3)
            fig, ax = plt.subplots(num_rows, num_cols, figsize=figsize)
        counter = 0
        ax_idx = []
        for num, mf in enumerate(np.arange(min_f, max_f + delta_f, delta_f)):
            if verbosity > 0:
                print('Estimate parameters for maximum fraction set to %.2f' % mf)
            m, beta, f = self._estimate_shape_scale(mf)
            if verbosity > 0:
                print('Estimated parameters for maximum fraction %.2f are\nm=\t%.3f\nbeta=\t%.5f' % (mf, m, beta))
                if verbosity > 1 and counter < num_cols * num_rows and not np.isnan(f):
                    self._plot_in_logspace(ax.reshape(-1)[counter], m=m, beta=beta, max_frac=mf)
                    ax.reshape(-1)[counter].set_title('Max fraction %.2f' % mf)
                    counter += 1

            fval.append(f)
            beta_est.append(beta)
            m_est.append(m)

        fval = np.asarray(fval)
        if np.all(np.isnan(fval)):
            self.m = None
            self.beta = None
            self.max_frac = None
            return

        mask = ~np.isnan(fval)
        # Take the lowest within the tolerance value
        idx = np.where(np.abs(np.nanmax(fval) - fval[mask]) < fval_accuracy)[0][0]
        self.beta = np.asarray(beta_est)[mask][idx]
        self.m = np.asarray(m_est)[mask][idx]
        self.max_frac = np.arange(min_f, max_f + delta_f, delta_f)[mask][idx]

        if verbosity > 1:
            if idx < num_cols * num_rows:
                for spine in ax.reshape(-1)[idx].spines.values():
                    spine.set_edgecolor('red')

            fig.suptitle('Model %s' % self.name)
            fig.tight_layout()
            if save_fig:
                directory = validate_dir('figures/data_models/jmak')
                fig.savefig('%s/%s_model_fitting.png' % (directory, save_prefix))
                plt.close('all')
            else:
                plt.show()

    def _plot_in_logspace(self, ax, m, beta, max_frac):
        dp = self.data_points.copy()
        dp[dp == 0] += np.finfo('float').eps
        dp[dp == 1] -= np.finfo('float').eps
        ax.scatter(
            np.log(self.time_points),
            np.log(np.log(1. / (1. - dp / max_frac))),
            label='Data'
        )
        scale = np.linspace(np.maximum(np.min(self.time_points) - 1, 1), np.max(self.time_points) + 1, 5)
        ax.plot(
            np.log(scale),
            [m * tlog + m * np.log(beta) for tlog in np.log(scale)],
            label='Estimation'
        )
        ax.legend(loc='lower right')

    def plot(self, figsize=(8, 7), save_fig=True, save_prefix=''):
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        self._plot_in_logspace(ax[0], self.m, self.beta, self.max_frac)
        ax[1].scatter(self.time_points, self.data_points, label='Data')
        time_scale = np.arange(np.max(self.time_points))
        ax[1].plot(time_scale, [self.repair_fraction(t) for t in time_scale], label='Estimation')

        ax[1].set_title('Over time')

        ax[1].legend(loc='lower right')
        fig.suptitle('Model prediction vs data')
        fig.tight_layout()
        if save_fig:
            directory = validate_dir('figures/data_models')
            fig.savefig('%s/%s_%s_model_estimation.png' % (directory, save_prefix, self.name))
            plt.close('all')
        else:
            plt.show()


class RegionModel:
    def __init__(self, time_points, data_points, name=''):
        # Assume that data and time points are reshaped such that the time and data per each model are grouped
        # together in a subarray
        if len(time_points.shape) != 2 or len(data_points.shape) != 2:
            raise ValueError('Time points and data points need to be reshaped such that each model has a'
                             ' corresponding row in array.')
        self.time_points = time_points
        self.data_points = data_points
        self.models = []
        self.name = name

    @staticmethod
    def iteration(time, data, n, min_f, max_f, delta_f, verbosity, figsize, save_fig, save_prefix):
        model = JMAK(time, data, name=n)
        model.estimate_parameters(
            min_f,
            max_f,
            delta_f,
            verbosity=verbosity,
            figsize=figsize,
            save_fig=save_fig,
            save_prefix=save_prefix
        )
        return model

    def fit_models(
            self,
            min_f,
            max_f=1.,
            delta_f=.01,
            names=None,
            verbosity=0,
            num_cpus=4,
            figsize=(8, 7),
            save_fig=True,
            save_prefix=''
    ):
        if num_cpus <= 1:
            for num, (t, d) in enumerate(zip(self.time_points, self.data_points)):
                self.models.append(self.iteration(
                    t, d, names[num] if names is not None else '',
                    min_f, max_f, delta_f, verbosity-1, figsize,
                    save_fig, '%s_%s' % (save_prefix, names[num] if names is not None else num))
                )
        else:
            num_cpus = np.minimum(multiprocessing.cpu_count() - 1, num_cpus)
            with multiprocessing.Pool(processes=num_cpus) as parallel:
                models = []
                for num, (t, d) in enumerate(zip(self.time_points, self.data_points)):
                    models.append(
                        parallel.apply_async(self.iteration,
                                             args=(t, d, names[num] if names is not None else '',
                                                   min_f, max_f, delta_f, verbosity-1, figsize, save_fig,
                                                   '%s_%s' % (save_prefix, names[num] if names is not None else num),))
                    )
                parallel.close()
                parallel.join()
                self.models = [ar.get() for ar in models]

    def get_total_repair_fraction(self, time_scale):
        all_rf = []
        for model in self.models:
            all_rf.append(model.repair_fraction_over_time(to_time=time_scale))
        return np.asarray(all_rf)

    def get_model_parameter(self, identifier='m', do_filter=True):
        if identifier == 'm':
            m_map = map(lambda x: x.m, self.models)
            if do_filter:
                return filter(lambda y: y is not None, m_map)
            else:
                return m_map
        elif identifier == 'beta':
            beta_map = map(lambda x: x.beta, self.models)
            if do_filter:
                return filter(lambda y: y is not None, beta_map)
            return beta_map
        elif identifier == 'max_frac':
            max_frac_map = map(lambda x: x.max_frac, self.models)
            if do_filter:
                return filter(lambda y: y is not None, max_frac_map)
            return max_frac_map
        elif identifier == 'name':
            name_map = map(lambda x: x.name if x.m is not None else None, self.models)
            if do_filter:
                return filter(lambda y: y is not None, name_map)
            return name_map
        else:
            raise ValueError('Invalid identifier')

    def load_models(self, file_path='', compare_chrom_list=None):
        file_path = transform_path(file_path)
        df = pd.read_csv(file_path, sep='\t')
        for entry, t, d, chrom in zip(df.iterrows(), self.time_points, self.data_points, compare_chrom_list):
            entry = entry[1]
            if ~np.isnan(entry['m']):
                if chrom not in entry['name']:
                    raise ValueError('Model values and data do not match')
            model = JMAK(t, d, m=entry['m'], max_frac=entry['max_frac'], beta=entry['beta'], name=entry['name'])
            self.models.append(model)

    def to_file(self, file_name=''):
        names = list(self.get_model_parameter('name', do_filter=False))
        m_list = list(self.get_model_parameter('m', do_filter=False))
        max_frac_list = self.get_model_parameter('max_frac', do_filter=False)
        beta_list = self.get_model_parameter('beta', do_filter=False)

        if not names:
            names = [''] * len(m_list)
        df = pd.DataFrame(zip(names, m_list, max_frac_list, beta_list), columns=['name', 'm', 'max_frac', 'beta'])
        directory = validate_dir('data/jmak')
        df.to_csv('%s/%s.csv' % (directory, file_name), sep='\t')

    def plot_parameter_histogram(
            self,
            color,
            bins=(30, 30),
            norm_gamma=.3,
            figsize=(4, 10),
            save_fig=True,
            save_prefix=''
    ):
        def plot_hist(param_x, param_y, a):
            hist, _, _ = np.histogram2d(param_y, param_x, bins=bins)
            heatmap = a.imshow(hist, norm=cls.PowerNorm(gamma=norm_gamma), cmap=color, origin='lower')
            a.set_xticks(np.concatenate([np.arange(0, hist.shape[0], 5), hist.shape[0]], axis=None))
            a.set_yticks(np.concatenate([np.arange(0, hist.shape[1], 5), hist.shape[1]], axis=None))

            a.set_xticklabels(
                ['%.2f' % l for l in np.linspace(
                    np.min(param_x),
                    np.max(param_x),
                    np.arange(0, hist.shape[0], 5).size + 1)])
            a.set_yticklabels(
                ['%.5f' % l for l in np.linspace(
                    np.min(param_y),
                    np.max(param_y),
                    np.arange(0, hist.shape[0], 5).size + 1)])

            return heatmap

        params_m = list(self.get_model_parameter('m'))
        params_beta = list(self.get_model_parameter('beta'))
        params_max_frac = list(self.get_model_parameter('max_frac'))
        fig, ax = plt.subplots(3, 1, figsize=figsize)

        hm_m_beta = plot_hist(params_m, params_beta, ax[0])
        hm_m_mf = plot_hist(params_m, params_max_frac, ax[1])
        hm_mf_beta = plot_hist(params_max_frac, params_beta, ax[2])

        ax[0].set_xlabel('m')
        ax[0].set_ylabel(r'$\beta$')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(hm_m_beta, cax=cax, orientation='vertical')

        ax[1].set_xlabel('m')
        ax[1].set_ylabel('Maximum fraction')
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(hm_m_mf, cax=cax, orientation='vertical')

        ax[2].set_xlabel('Maximum fraction')
        ax[2].set_ylabel(r'$\beta$')
        divider = make_axes_locatable(ax[2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(hm_mf_beta, cax=cax, orientation='vertical')

        fig.suptitle('%s\nParameter distribution' % self.name)
        fig.tight_layout()

        if save_fig:
            directory = validate_dir('figures/data_models')
            fig.savefig('%s/%s_%s_model_estimation.png' % (directory, save_prefix, self.name))
            plt.close('all')
        else:
            plt.show()

    def plot_parameter_with_gradient(
            self,
            cgradient,
            size_power=4,
            size_scaling=200,
            figsize=(8, 7),
            cmap='seismic',
            alpha=.7,
            num_handles=6,
            power_norm=1,
            add_beta_legend=False,
            order_cgradient=True,
            data_mask=None,
            m_range=None,
            save_fig=True,
            save_prefix=False
    ):
        m = np.asarray(list(self.get_model_parameter('m')))
        mf = np.asarray(list(self.get_model_parameter('max_frac')))
        beta = (np.asarray(list(self.get_model_parameter('beta'))) * size_scaling) ** size_power

        if data_mask is None:
            data_mask = np.ones(m.size, dtype='bool')
        mask = np.logical_and(~np.isnan(m), np.logical_and(~np.isnan(beta), ~np.isnan(mf)))
        if m_range is not None:
            mask[np.logical_or(m < m_range[0], m > m_range[1])] = False
        mask = np.logical_and(data_mask, mask)
        m = m[mask]
        mf = mf[mask]
        beta = beta[mask]
        cgradient = cgradient[mask]
        if m.size == 0 or mf.size == 0 or beta.size == 0:
            return

        if order_cgradient:
            bins = frequency_bins(cgradient, 50)
            cgrad = np.digitize(cgradient, bins=bins)
        else:
            cgrad = cgradient
            bins = cgradient

        fig = plt.figure(figsize=figsize)
        scatter = plt.scatter(
            m,
            mf,
            c=cgrad,
            s=beta,
            cmap=cmap,
            norm=cls.PowerNorm(power_norm),
            alpha=alpha
        )
        plt.xlabel('m')
        plt.ylabel('Maximum fraction')
        plt.title('Parameter distribution %s' % self.name)
        label_idx = np.linspace(np.min(cgrad), np.max(cgrad), 6, dtype='int')
        cbar = plt.colorbar(scatter, ticks=label_idx)
        cbar.ax.set_yticklabels(['%.2f' % label for label in bins[label_idx - 1]])

        if add_beta_legend:
            handles, labels = scatter.legend_elements(
                'sizes',
                num=num_handles,
                func=lambda x: np.power(x, 1./size_power) / size_scaling
            )
            plt.legend(
                handles,
                labels,
                title=r'$\beta$',
                handletextpad=2,
                labelspacing=1.5,
                frameon=False,
                loc='center left',
                bbox_to_anchor=(1.4, 0.5)
            )
            fig.tight_layout(rect=[0, 0, 0.95, 1])
        else:
            fig.tight_layout()
        if save_fig:
            try:
                directory = validate_dir('figures/grad_colour')
                fig.savefig('%s/%s_%s_model_parameter_gradient.png' % (directory, save_prefix, self.name))
                plt.close('all')
            except:
                plt.close('all')
        else:
            plt.show()


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
            min_m=.4,
            max_m=4.5,
            min_mf=.4,
            max_mf=1.,
            min_beta=8e-3,
            max_beta=3.5e-2,
            val_frac=.2,
            rm_percentile=5.,
            random_state=None,
            num_param_values=100
    ):
        if len(rmodel.models) != len(bio_data):
            raise ValueError('The number of region models must be equal to the number of biological data points.')
        self.rmodel = rmodel
        self.map_bio = None
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
        if self.discretise_bio:
            self.bio_bins = frequency_bins(self.all_bio_data, num_bins)
            self.all_bio_data = np.digitize(self.all_bio_data, self.bio_bins)
        if randomise:
            np.random.shuffle(self.all_bio_data)

        m_shape, self.m_mean, self.m_std = ParameterMap.normalise(m_shape[param_mask])
        max_frac, self.max_frac_mean, self.max_frac_std = ParameterMap.normalise(max_frac[param_mask])
        beta, self.beta_mean, self.beta_std = ParameterMap.normalise(beta[param_mask])

        self.all_data_params = np.asarray([m_shape, max_frac, beta]).T
        self.data_params, self.data_params_val, self.bio_data, self.bio_data_val = None, None, None, None
        self.reshuffle()

        self.map_m = np.linspace(min_m, max_m, num_param_values)
        self.map_max_frac = np.linspace(min_mf, max_mf, num_param_values)
        self.map_beta = np.linspace(min_beta, max_beta, num_param_values)
        self.map_m, self.map_max_frac, self.map_beta = np.meshgrid(self.map_m, self.map_max_frac, self.map_beta)

        self.map_m = np.linspace(min_m, max_m, num_param_values)
        self.map_max_frac = np.linspace(min_mf, max_mf, num_param_values)
        self.map_beta = np.linspace(min_beta, max_beta, num_param_values)
        self.map_m, self.map_max_frac, self.map_beta = np.meshgrid(self.map_m, self.map_max_frac, self.map_beta)

        # Rescale parameters into the right space where it was trained on
        self.map_param = np.asarray([
            ((self.map_m - self.m_mean) / self.m_std).reshape(-1),
            ((self.map_max_frac - self.max_frac_mean) / self.max_frac_std).reshape(-1),
            ((self.map_beta - self.beta_mean) / self.beta_std).reshape(-1)
        ]).T

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

    @staticmethod
    def error(time_sample, real_data, est_mean, est_var=None):
        time_sample = np.asarray(time_sample)
        s = (np.asarray(est_mean) * 100 - np.asarray(real_data) * 100)**2
        if est_var is None:
            return np.mean(s)
        else:
            s = np.sum(s)
            return (s + np.sum(est_var)) / float(len(real_data))

    def plot_parameter_map(
            self,
            plotted_dp=100,
            levels=15,
            figsize=(8, 7),
            verbosity=0,
            save_fig=True,
            save_prefix=''
    ):
        pca = PCA(n_components=2)
        map_param_pca = pca.fit(self.map_param).transform(self.map_param).T
        if verbosity > 0:
            print(
                'Explained parameter ratio:\tPC1 %.2f\tPC2: %.2f' %
                (pca.explained_variance_ratio_[0],
                 pca.explained_variance_ratio_[1])
            )
        params_val_pca = pca.transform(self.data_params_val).T

        plt.figure(figsize=figsize)
        if self.discretise_bio:
            map_bio = np.argmax(self.map_bio, axis=1)
        else:
            map_bio = self.map_bio
        map_contour = plt.tricontourf(*map_param_pca, map_bio, levels, cmap='seismic', alpha=.7)
        plt.tricontour(*map_param_pca, map_bio, levels, linewidths=0.5, colors='k')

        size = np.minimum(plotted_dp, len(self.bio_data_val))
        idx = np.random.choice(len(self.bio_data_val), size=size, replace=False)
        plt.scatter(
            *params_val_pca.T[idx].T,
            c=self.bio_data_val[idx],
            cmap='seismic',
            edgecolor='white',
            norm=cls.Normalize(vmin=np.min(map_bio), vmax=np.max(map_bio))
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

    @staticmethod
    def det_parameter_mapping(xn, *args):
        pass

    def learn(
            self,
            num_cpus=1,
            estimate_time=140,
            verbosity=0,
            figsize=(8, 7),
            hist_bins=100,
            save_fig=True,
            save_prefix=''
    ):
        pass

    def estimate(self, new_bio, time_scale=140, discretise_nb=True):
        est_m_list, est_max_frac_list, est_beta_list, prediction = [], [], [], []
        for nb in new_bio:
            if self.discretise_bio:
                if discretise_nb:
                    nb_class = np.digitize(nb, self.bio_bins)
                else:
                    nb_class = nb
                idc = np.where(self.map_bio[:, nb_class] > 1. / self.map_bio.shape[1])[0]
                mask = np.zeros(self.map_bio.shape[0], dtype='bool')
                mask[idc] = True
                weights = self.map_bio[:, nb_class][idc] / np.sum(self.map_bio[:, nb_class][idc])
            else:
                distances = np.abs(self.map_bio - nb)
                mask = distances < np.sqrt(self.learnt_var)
                weights = 1 - (distances / np.sqrt(self.learnt_var))[mask]
                weights /= np.sum(weights)

            est_m, est_max_frac, est_beta = (
                self.map_m.reshape(-1)[mask],
                self.map_max_frac.reshape(-1)[mask],
                self.map_beta.reshape(-1)[mask]
            )
            est_m = est_m.dot(weights)
            est_max_frac = est_max_frac.dot(weights)
            est_beta = est_beta.dot(weights)

            est_m_list.append(est_m)
            est_max_frac_list.append(est_max_frac)
            est_beta_list.append(est_beta)
            tp_model = JMAK(
                time_points=None,
                data_points=None,
                m=est_m,
                max_frac=est_max_frac,
                beta=est_beta,
                name='Temp'
            )
            prediction.append(tp_model.repair_fraction_over_time(time_scale).T)
            # prediction.append(tp_model.repair_fraction_over_time(time_scale).dot(weights))

        return prediction, est_m_list, est_max_frac_list, est_beta_list


class BayesianParameterMap(ParameterMap):
    def __init__(
            self,
            rmodel,
            bio_data,
            kernel_func_type,
            discretise_bio=True,
            randomise=False,
            num_bins=50,
            kernel_search_step=5e-4,
            kernel_search_thresh=1e-3,
            kernel_search_verbosity=0,
            kernel_param_min=.001,
            kernel_scaling_init=5.,
            kernel_random_ratio=.3,
            kernel_max_iter=10000,
            noise_scaling_init=2.,
            min_m=.4,
            max_m=4.5,
            min_mf=.4,
            max_mf=1.,
            min_beta=8e-3,
            max_beta=3.5e-2,
            num_param_values=100,
            val_frac=.2,
            rm_percentile=5.,
            num_cpus=1.,
            random_state=None,
            do_estimate_hp=True
    ):
        if len(rmodel.models) == 1:
            raise ValueError('The region model must contain at least two JMAK models.')
        if discretise_bio:
            raise ValueError('The Gaussian implementation does not allow a classification. Use the kNN model instead.')
        super().__init__(
            rmodel,
            bio_data,
            discretise_bio=discretise_bio,
            num_bins=num_bins,
            randomise=randomise,
            min_m=min_m,
            max_m=max_m,
            min_mf=min_mf,
            max_mf=max_mf,
            min_beta=min_beta,
            max_beta=max_beta,
            num_param_values=num_param_values,
            rm_percentile=rm_percentile,
            val_frac=val_frac,
            random_state=random_state
        )
        self.num_cpus = num_cpus
        if kernel_func_type == 'eqk':
            kernel_func = exponential_quadratic_kernel
            if do_estimate_hp:
                kernel_param, self.noise = self._optimise_eqk_param(
                    kernel_search_step,
                    kernel_search_thresh,
                    kernel_scaling_init,
                    noise_scaling_init,
                    kernel_param_min,
                    kernel_search_verbosity,
                    random_ratio=kernel_random_ratio,
                    max_iter=kernel_max_iter
                )
            else:
                # Dummy parameters for testing
                self.noise = noise_scaling_init
                kernel_param = np.asarray([4, 2, 1])
        elif kernel_func_type == 'gaussian':
            kernel_func = gaussian_kernel
            if do_estimate_hp:
                kernel_param, self.noise = self._optimise_gaussian_param(
                    kernel_search_step,
                    kernel_search_thresh,
                    kernel_scaling_init,
                    noise_scaling_init,
                    kernel_param_min,
                    kernel_search_verbosity
                )
            else:
                # Dummy parmaeters for testing
                self.noise = noise_scaling_init
                kernel_param = 2.
        else:
            raise ValueError('Passed kernel function type invalid.')

        self.kernel = ParameterMap.KernelWrapper(kernel_func, kernel_param)
        self.C = None
        self.inv_C = None

    @staticmethod
    def _calc_c(data, kernel, kernel_param, noise, num_cpus):
        if num_cpus < 2:
            C = np.asarray([
                kernel(x_i, x_j, *kernel_param)
                for x_i in data for x_j in data
            ]).reshape(data.shape[0], data.shape[0])
            C += np.eye(C.shape[0]) * noise
        else:
            num_cpus = np.minimum(num_cpus, multiprocessing.cpu_count() - 1)
            with multiprocessing.Pool(processes=num_cpus) as parallel:
                result = parallel.starmap(
                    kernel,
                    ([*d, *theta]
                     for d, theta in zip(product(data, data),
                                         repeat(kernel_param, (data.shape[0] * data.shape[0])))
                     )
                )
                parallel.close()
                parallel.join()
            C = np.asarray(result).reshape(data.shape[0], data.shape[0])

        if np.any(np.isnan(C)):
            warnings.warn('NaN values detected in Gramm matrix. Replace them by 0.')
            C = np.nan_to_num(C, nan=0.)
        try:
            inv_C = np.linalg.pinv(C)
        except np.linalg.LinAlgError:
            warnings.warn('Could not create pseudo inverse. Return None.')
            return None, None

        return C, inv_C

    @staticmethod
    def jac(sub_jac, inv_C, bio_d):
        return (
                -.5 * np.trace(inv_C.dot(sub_jac))
                + 0.5 * bio_d.dot(inv_C.dot(sub_jac.dot(bio_d.T)))
        )

    @staticmethod
    def _subjac_precision(size, precision):
        return - np.eye(size) * precision**(-2)

    def _optimise_eqk_param(
            self,
            step_size,
            thresh,
            scaling_init,
            noise_scaling_init,
            min_param,
            verbosity,
            random_ratio=.3,
            max_iter=10000
    ):
        theta_old = np.ones(4) * 999
        s = np.ones(4) * scaling_init
        s[-1] = noise_scaling_init
        theta_new = np.random.randn(4) + s
        counter = 0
        while np.linalg.norm(theta_old - theta_new) > thresh and counter < max_iter:
            counter += 1
            if verbosity > 0:
                start = time.time()
            theta_old = theta_new.copy()
            idc = np.random.choice(len(self.data_params), size=int(len(self.data_params) * random_ratio))
            C_theta, inv_C_theta = BayesianParameterMap._calc_c(
                self.data_params[idc],
                exponential_quadratic_kernel,
                theta_old[:3],
                1. / theta_old[3],
                self.num_cpus
            )
            if C_theta is None or inv_C_theta is None:
                theta_old = np.ones(4) * 999
                continue

            if self.num_cpus < 2:
                sub_jac_theta1 = np.asarray(
                    [jac_eqk_theta_1(x_i, x_j, theta_old[1])
                     for x_i in self.data_params[idc] for x_j in self.data_params[idc]]
                ).reshape(len(idc), len(idc))

                sub_jac_theta2 = np.asarray(
                    [jac_eqk_theta_2(x_i, x_j, *theta_old[:2])
                     for x_i in self.data_params[idc] for x_j in self.data_params[idc]]
                ).reshape(len(idc), len(idc))

                sub_jac_theta3 = np.asarray(
                    [jac_eqk_theta_3(x_i, x_j) for x_i in self.data_params[idc] for x_j in self.data_params[idc]]
                ).reshape(len(idc), len(idc))

            else:
                num_cpus = np.minimum(multiprocessing.cpu_count() - 1, self.num_cpus)
                with multiprocessing.Pool(processes=num_cpus) as parallel:
                    sub_jac_theta1 = parallel.starmap(
                        jac_eqk_theta_1,
                        ([*d, theta] for d, theta in
                         zip(
                            product(self.data_params[idc], self.data_params[idc]),
                            repeat(theta_old[1], (len(idc) * len(idc)))
                        ))
                    )

                    sub_jac_theta2 = parallel.starmap(
                        jac_eqk_theta_2,
                        ([*d, *theta] for d, theta in
                         zip(
                             product(self.data_params[idc], self.data_params[idc]),
                             repeat(theta_old[:2], (len(idc) * len(idc)))
                        ))
                    )

                    sub_jac_theta3 = parallel.starmap(
                        jac_eqk_theta_3,
                        product(self.data_params[idc], self.data_params[idc])
                    )

                    sub_jac_theta1 = np.asarray(sub_jac_theta1).reshape(
                        len(idc), len(idc))
                    sub_jac_theta2 = np.asarray(sub_jac_theta2).reshape(
                        len(idc), len(idc))
                    sub_jac_theta3 = np.asarray(sub_jac_theta3).reshape(
                        len(idc), len(idc))
            sub_jac_precision = self._subjac_precision(len(idc), theta_old[3])
            jact1 = self.jac(sub_jac_theta1, inv_C_theta, self.bio_data[idc])
            jact2 = self.jac(sub_jac_theta2, inv_C_theta, self.bio_data[idc])
            jact3 = self.jac(sub_jac_theta3, inv_C_theta, self.bio_data[idc])
            jacp = self.jac(sub_jac_precision, inv_C_theta, self.bio_data[idc])
            jac_theta = np.asarray([jact1, jact2, jact3, jacp])
            theta_new = np.maximum(theta_old + step_size * jac_theta, min_param)

            if verbosity > 0:
                print('Time taken: %.2f sec' % (time.time() - start))
                print('Diff: %s' % np.linalg.norm(theta_old - theta_new))
                print('New parameters: %s' % theta_new)

        return theta_new[:3], 1. / theta_new[3]

    def _optimise_gaussian_param(
            self,
            step_size,
            thresh,
            scaling_init,
            noise_scaling_init,
            min_param,
            verbosity,
            random_ratio=.3,
            max_iter=10000
    ):
        sigma_sq_old = np.ones(2) * 999
        s = np.ones(2) * scaling_init
        s[-1] = noise_scaling_init
        sigma_sq_new = np.random.randn(2) + s
        counter = 0
        while np.linalg.norm(sigma_sq_old - sigma_sq_new) > thresh and counter < max_iter:
            counter += 1
            sigma_sq_old = sigma_sq_new.copy()
            idc = np.random.choice(len(self.data_params), size=int(len(self.data_params) * random_ratio))
            C_sigma, inv_C_sigma = BayesianParameterMap._calc_c(
                self.data_params[idc],
                gaussian_kernel,
                sigma_sq_old[0],
                1. / sigma_sq_old[1],
                self.num_cpus
            )
            if C_sigma is None or inv_C_sigma is None:
                continue
            sub_jac_sigma = np.asarray(
                [jac_gaussian_kernel(x_i, x_j, sigma_sq_old[0])
                 for x_i in self.data_params[idc] for x_j in self.data_params[idc]]
            ).reshape(len(idc), len(idc))
            sub_jac_precision = self._subjac_precision(len(idc), sigma_sq_old[1])
            jac_sigma = self.jac(sub_jac_sigma, inv_C_sigma)
            jac_precision = self.jac(sub_jac_precision, inv_C_sigma)
            jac_param = np.asarray([jac_sigma, jac_precision])
            sigma_sq_new = np.maximum(sigma_sq_old + step_size * jac_param, min_param)
            if verbosity > 0:
                print('Diff %s' % np.linalg.norm(sigma_sq_old - sigma_sq_new))
                print('New parameter %s' % sigma_sq_new)

        return sigma_sq_new[0], 1. / sigma_sq_new[1]

    def _set_gramm(self):
        self.C, self.inv_C = BayesianParameterMap._calc_c(
            self.data_params,
            self.kernel.func,
            self.kernel.params,
            self.noise,
            self.num_cpus
        )

    @staticmethod
    def det_parameter_mapping(xn, y_data, kernel, weighted_t, inv_C, noise):
        k = np.asarray([kernel(x_j, xn) for x_j in y_data])
        m = k.dot(weighted_t)
        v = kernel(xn, xn) + noise - k.dot(inv_C).dot(k.T)
        return m, v

    def learn(
            self,
            num_cpus=1,
            estimate_time=140,
            verbosity=0,
            figsize=(8, 7),
            hist_bins=100,
            save_fig=True,
            save_prefix=''
    ):
        self._set_gramm()

        mean, var = [], []
        weighted_t = self.inv_C.dot(self.bio_data)
        if num_cpus < 2:
            for xn in self.map_param:
                m, v = self.det_parameter_mapping(xn, self.data_params, self.kernel,
                                                  weighted_t, self.inv_C, self.noise)
                mean.append(m)
                var.append(v)
        else:
            num_cpus = np.minimum(multiprocessing.cpu_count() - 1, num_cpus)
            with multiprocessing.Pool(processes=num_cpus) as parallel:
                results = []
                for xn in self.map_param:
                    results.append(parallel.apply_async(
                        self.det_parameter_mapping, (xn, self.data_params, self.kernel,
                                                     weighted_t, self.inv_C, self.noise,)
                    ))
                parallel.close()
                parallel.join()
                results = [r.get() for r in results]
                mean, var = zip(*results)

        self.map_bio = np.asarray(mean)
        self.learnt_var = np.asarray(var)
        predictions, _, _, _ = self.estimate(new_bio=self.bio_data_val, time_scale=estimate_time)
        val_error = []
        for (m, mf, beta), p in zip(self.data_params_val, predictions):
            m = m * self.m_std + self.m_mean
            mf = mf * self.max_frac_std + self.max_frac_mean
            beta = beta * self.beta_std + self.beta_mean
            best_estimation = JMAK(None, None, m=m, max_frac=mf, beta=beta
                                   ).repair_fraction_over_time(to_time=estimate_time)
            error = self.error(np.arange(estimate_time), best_estimation, p, None)
            val_error.append(error)
        if verbosity > 0:
            print('Model: %s\tAverage validation error: %.2f' % (self.rmodel.name, np.mean(val_error)))
            if verbosity > 1:
                plt.figure(figsize=(8, 7))
                plt.hist(val_error, bins=hist_bins)
                plt.title('Validation error histogram: %s' % self.rmodel.name)
                if save_fig:
                    directory = validate_dir('figures/predict_models')
                    plt.savefig('%s/%s_%s_val_err.png' % (directory, save_prefix, self.rmodel.name))
                    plt.close('all')
                else:
                    plt.show()

        return val_error


class NNParameterMap(ParameterMap):
    def __init__(
            self,
            rmodel,
            bio_data,
            discretise_bio=True,
            num_bins=50,
            randomise=False,
            layer_sizes=[64, 32, 8],
            step_size=1e-3,
            num_epocs=200,
            k_fold=5,
            min_m=.4,
            max_m=4.5,
            min_mf=.4,
            max_mf=1.,
            min_beta=8e-3,
            max_beta=3.5e-2,
            num_param_values=100,
            val_frac=.2,
            rm_percentile=5.,
            num_cpus=4,
            random_state=None,
    ):
        if len(rmodel.models) == 1:
            raise ValueError('The region model must contain at least two JMAK models.')

        super().__init__(
            rmodel,
            bio_data,
            discretise_bio=discretise_bio,
            randomise=randomise,
            num_bins=num_bins,
            min_m=min_m,
            max_m=max_m,
            min_mf=min_mf,
            max_mf=max_mf,
            min_beta=min_beta,
            max_beta=max_beta,
            num_param_values=num_param_values,
            val_frac=val_frac,
            rm_percentile=rm_percentile,
            random_state=random_state
        )

        tf.config.threading.set_intra_op_parallelism_threads(num_cpus)
        tf.config.threading.set_inter_op_parallelism_threads(num_cpus)
        if self.discretise_bio:
            # Allow for one greater as largest class in bio data as this allows to account for everything that is
            # larger than so far seen data
            self.bio_data = tf.one_hot(self.bio_data, depth=int(np.max(self.all_bio_data)) + 1).numpy()
        self.num_epochs = num_epocs
        self.layer_sizes = layer_sizes
        self.step_size = step_size
        self.k_fold = k_fold
        self.nn = None

    @staticmethod
    def _build_nn(input_data, layer_sizes, step_size, num_output_neurons=1, do_classify=False):
        layer_list = []
        for num, ls in enumerate(layer_sizes):
            layer_list.append(layers.Dense(ls, activation='relu'))
            if num == 0:
                layer_list.append(layers.Dropout(.7))
            elif num == len(layer_sizes) - 1:
                layer_list.append(layers.Dropout(.5))
        if do_classify:
            layer_list.append(layers.Dense(num_output_neurons, activation='relu'))
        else:
            layer_list.append(layers.Dense(num_output_neurons, activation='relu'))

        nn = keras.Sequential(layer_list)
        nn.compile(
            loss='mean_squared_error' if not do_classify else tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adagrad(step_size),
            metrics=['accuracy', 'mean_squared_error']
        )
        return nn

    def _train_nn(self, batch_size=30, figsize=(8, 7), verbosity=0, save_fig=True, save_prefix=''):
        kfold = KFold(n_splits=self.k_fold, shuffle=True)

        hist_loss = np.zeros((self.k_fold, self.num_epochs))
        hist_loss_val = np.zeros((self.k_fold, self.num_epochs))
        hist_acc = np.zeros((self.k_fold, self.num_epochs))
        hist_acc_val = np.zeros((self.k_fold, self.num_epochs))
        nn_list = []
        for num, (train_t, test_t) in enumerate(kfold.split(self.data_params, self.bio_data)):
            if verbosity > 0:
                print('Fold number %s' % (num + 1))
            nn = self._build_nn(
                self.data_params,
                self.layer_sizes,
                self.step_size,
                num_output_neurons=1 if not self.discretise_bio else self.bio_data.shape[1],
                do_classify=self.discretise_bio
            )
            nn_list.append(nn)

            h = nn.fit(
                self.data_params[train_t],
                self.bio_data[train_t],
                validation_data=(
                    self.data_params[test_t],
                    self.bio_data[test_t].reshape(-1, 1) if not self.discretise_bio else self.bio_data[test_t]
                ),
                epochs=self.num_epochs,
                batch_size=batch_size,
                verbose=0 if verbosity < 2 else 1
            )

            if self.discretise_bio:
                hist_acc[num] = np.asarray(h.history['accuracy'])
                hist_acc_val[num] = np.asarray(h.history['val_accuracy'])
            hist_loss[num] = np.asarray(h.history['loss'])
            hist_loss_val[num] = np.asarray(h.history['val_loss'])

        self.nn = nn_list[np.random.choice(self.k_fold)]
        if self.discretise_bio:
            self.nn = tf.keras.Sequential([self.nn, layers.Softmax()])
        if verbosity > 1:
            fig = plt.figure(figsize=figsize)
            host = host_subplot(111, axes_class=AA.Axes, figure=fig)

            host.plot(
                np.arange(self.num_epochs),
                np.mean(hist_loss, axis=0),
                color='tab:blue',
                label='MSE' if not self.discretise_bio else 'Categorical CE'
            )
            host.fill_between(
                np.arange(self.num_epochs),
                np.mean(hist_loss, axis=0) - np.var(hist_loss, axis=0),
                np.mean(hist_loss, axis=0) + np.var(hist_loss, axis=0),
                color='tab:blue',
                alpha=.2
            )

            host.plot(
                np.arange(self.num_epochs),
                np.mean(hist_loss_val, axis=0),
                color='cyan',
                label='Val MSE' if not self.discretise_bio else 'Val Categorical CE'
            )
            host.fill_between(
                np.arange(self.num_epochs),
                np.mean(hist_loss_val, axis=0) - np.var(hist_loss_val, axis=0),
                np.mean(hist_loss_val, axis=0) + np.var(hist_loss_val, axis=0),
                color='cyan',
                alpha=.2
            )

            if self.discretise_bio:
                par1 = host.twinx()
                par1.axis['right'].toggle(all=True)
                par1.set_ylabel('Accuracy')
                par1.set_ylim((0., 1.))
                par1.plot(np.arange(self.num_epochs), np.mean(hist_acc, axis=0), color='tab:orange', label='Accuracy')
                par1.fill_between(
                    np.arange(self.num_epochs),
                    np.mean(hist_acc, axis=0) - np.var(hist_acc, axis=0),
                    np.mean(hist_acc, axis=0) + np.var(hist_acc, axis=0),
                    color='tab:orange',
                    alpha=.2
                )
                par1.plot(
                    np.arange(self.num_epochs),
                    np.mean(hist_acc_val, axis=0),
                    color='darkgoldenrod',
                    label='Val Accuracy'
                )
                par1.fill_between(
                    np.arange(self.num_epochs),
                    np.mean(hist_acc_val, axis=0) - np.var(hist_acc_val, axis=0),
                    np.mean(hist_acc_val, axis=0) + np.var(hist_acc_val, axis=0),
                    color='darkgoldenrod',
                    alpha=.2
                )

            host.set_title('Learning process: %s' % self.rmodel.name)
            host.set_ylabel('Mean squared error' if not self.discretise_bio else 'Categorical CE')
            host.set_xlabel('Epoch')
            host.set_xlim((0, self.num_epochs))
            host.legend(loc='right')
            if save_fig:
                directory = validate_dir('figures/predict_models')
                plt.savefig('%s/%s_%s_nn_learning.png' % (directory, save_prefix, self.rmodel.name))
                plt.close('all')
            else:
                plt.show()

    def learn(
            self,
            num_cpus=1,
            batch_size=30,
            estimate_time=140,
            verbosity=0,
            figsize=(8, 7),
            hist_bins=100,
            save_fig=True,
            save_prefix=''
    ):
        # Use only single variance value
        if verbosity > 0:
            print('Train NN')
        self.learnt_var = None
        self._train_nn(
            batch_size=batch_size,
            figsize=figsize, 
            verbosity=verbosity,
            save_fig=save_fig,
            save_prefix=save_prefix
        )
        if verbosity > 0:
            print('Predict parameter map')

        self.map_bio = self.nn.predict(self.map_param, batch_size=batch_size)
        if not self.discretise_bio:
            self.map_bio = self.map_bio.reshape(-1)
        if verbosity > 0:
            print('Validate learnt parameter map')
        predictions, _, _, _ = self.estimate(new_bio=self.bio_data_val, time_scale=estimate_time, discretise_nb=False)
        val_error = []
        for (m, mf, beta), p in zip(self.data_params_val, predictions):
            m = m * self.m_std + self.m_mean
            mf = mf * self.max_frac_std + self.max_frac_mean
            beta = beta * self.beta_std + self.beta_mean
            best_estimation = JMAK(None, None, m=m, max_frac=mf, beta=beta
                                   ).repair_fraction_over_time(to_time=estimate_time)

            error = self.error(np.arange(estimate_time), best_estimation, p, None)
            val_error.append(error)
        if verbosity > 0:
            print('Model: %s\tAverage validation error: %.2f' % (self.rmodel.name, np.mean(val_error)))
            if verbosity > 1:
                plt.figure(figsize=(8, 7))
                plt.hist(val_error, bins=hist_bins)
                plt.title('Validation error histogram: %s' % self.rmodel.name)
                if save_fig:
                    directory = validate_dir('figures/predict_models')
                    plt.savefig('%s/%s_%s_val_err.png' % (directory, save_prefix, self.rmodel.name))
                    plt.close('all')
                else:
                    plt.show()

        return val_error

