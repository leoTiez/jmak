import warnings

import numpy as np
import multiprocessing

import pandas as pd
import statsmodels.api as smapi
import matplotlib.pyplot as plt
import matplotlib.colors as cls
from mpl_toolkits.axes_grid1 import make_axes_locatable, host_subplot
from matplotlib.ticker import StrMethodFormatter

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

    def repair_derivative(self, time):
        if self.m is None or self.max_frac is None or self.beta is None:
            raise ValueError(
                'Model parameters have not been set.\n'
                'Run estimate_parameters to fit the model to the data, or set the parameters manually'
            )

        return (
                self.max_frac * self.m * self.beta**self.m * time**(self.m - 1)
                * (1 - self.repair_fraction(time) / self.max_frac)
        )

    def repair_fraction_over_time(self, to_time):
        return np.asarray([self.repair_fraction(t) for t in np.arange(to_time)])

    def repair_derivative_over_time(self, to_time):
        return np.asarray([self.repair_derivative(t) for t in np.arange(to_time)])

    def _estimate_shape_scale(self, max_frac):
        if np.any(max_frac < self.data_points):
            return np.nan, np.nan, np.nan, np.nan
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
                result.rsquared_adj,  # R2
                result.fvalue  # fvalue
            )
        else:
            return np.nan, np.nan, np.nan, np.nan

    def estimate_parameters(
            self,
            min_f,
            max_f=1.,
            delta_f=.01,
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

        rsquared = []
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
            m, beta, rs, f = self._estimate_shape_scale(mf)
            if verbosity > 0:
                print('Estimated parameters for maximum fraction %.2f are\nm=\t%.3f\nbeta=\t%.5f' % (mf, m, beta))
                if verbosity > 1 and counter < num_cols * num_rows and not np.isnan(rs):
                    self._plot_in_logspace(ax.reshape(-1)[counter], m=m, beta=beta, max_frac=mf)
                    ax.reshape(-1)[counter].set_title('Max fraction %.2f' % mf)
                    ax_idx.append(num)
                    counter += 1

            rsquared.append(rs)
            beta_est.append(beta)
            m_est.append(m)

        rsquared = np.asarray(rsquared)
        if np.all(np.isnan(rsquared)):
            self.m = None
            self.beta = None
            self.max_frac = None
            return

        idx = np.nanargmax(rsquared)
        self.beta = np.asarray(beta_est)[idx]
        self.m = np.asarray(m_est)[idx]
        self.max_frac = np.arange(min_f, max_f + delta_f, delta_f)[idx]

        if verbosity > 0:
            print('Best maximum fraction is %.2f' % self.max_frac)

        if verbosity > 1:
            if idx in ax_idx:
                ai = ax_idx.index(idx)
                for spine in ax.reshape(-1)[ai].spines.values():
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
        if model.m is None or np.isnan(model.m):
            if verbosity >= 0:
                warnings.warn('Could not determine KJMA parameters.')
                print('Name: ', n)
                print('Data: ', data)
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

    def get_total_repair_derivative(self, time_scale):
        all_rd = []
        for model in self.models:
            all_rd.append(model.repair_derivative_over_time(to_time=time_scale))
        return np.asarray(all_rd)

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

        params_m = np.asarray(list(self.get_model_parameter('m')))
        params_beta = 1./np.asarray(list(self.get_model_parameter('beta')))
        params_max_frac = np.asarray(list(self.get_model_parameter('max_frac')))
        mask = ~np.isnan(params_beta)
        params_m = params_m[mask]
        params_beta = params_beta[mask]
        params_max_frac = params_max_frac[mask]
        fig, ax = plt.subplots(3, 1, figsize=figsize)

        hm_m_beta = plot_hist(params_m, params_beta, ax[0])
        hm_m_mf = plot_hist(params_m, params_max_frac, ax[1])
        hm_mf_beta = plot_hist(params_max_frac, params_beta, ax[2])

        ax[0].set_xlabel('m')
        ax[0].set_ylabel(r'$\tau$')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(hm_m_beta, cax=cax, orientation='vertical')

        ax[1].set_xlabel('m')
        ax[1].set_ylabel('Maximum fraction')
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(hm_m_mf, cax=cax, orientation='vertical')

        ax[2].set_xlabel('Maximum fraction')
        ax[2].set_ylabel(r'$\tau$')
        divider = make_axes_locatable(ax[2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(hm_mf_beta, cax=cax, orientation='vertical')

        ax[0].set_yticklabels(['%.2f' % float(y._text) for y in ax[0].yaxis.get_ticklabels()])
        ax[1].set_yticklabels(['%.2f' % float(y._text) for y in ax[1].yaxis.get_ticklabels()])
        ax[2].set_yticklabels(['%.2f' % float(y._text) for y in ax[2].yaxis.get_ticklabels()])
        ax[0].set_xticklabels(['%.2f' % float(x._text) for x in ax[0].xaxis.get_ticklabels()])
        ax[1].set_xticklabels(['%.2f' % float(x._text) for x in ax[1].xaxis.get_ticklabels()])
        ax[2].set_xticklabels(['%.2f' % float(x._text) for x in ax[2].xaxis.get_ticklabels()])
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
            title_add='',
            add_beta_legend=False,
            order_cgradient=True,
            data_mask=None,
            m_range=None,
            save_fig=True,
            save_prefix=False
    ):
        m = np.asarray(list(self.get_model_parameter('m')))
        mf = np.asarray(list(self.get_model_parameter('max_frac')))
        beta = np.minimum((np.asarray(list(self.get_model_parameter('beta'))) * size_scaling) ** size_power, 500)

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
        if m_range is not None:
            plt.xlim(m_range)
        plt.xlabel('m', fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylabel(r'$\theta$', fontsize=20)
        plt.title('%s\n%s' % (self.name, title_add), fontsize=30)
        label_idx = np.linspace(np.min(cgrad), np.max(cgrad), 6, dtype='int')
        cbar = plt.colorbar(scatter, ticks=label_idx)
        cbar.ax.set_yticklabels(['%.2f' % label for label in bins[label_idx - 1]], fontsize=16)

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


