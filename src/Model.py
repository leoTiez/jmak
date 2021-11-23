import numpy as np
import pandas as pd
import statsmodels.api as smapi
import matplotlib.pyplot as plt
import matplotlib.colors as cls
from mpl_toolkits.axes_grid1 import make_axes_locatable
from abc import ABC
import multiprocessing
import warnings
from sklearn.model_selection import train_test_split

from src.Utils import validate_dir
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
        def frequency_bins(x, nbin):
            nlen = len(x)
            return np.interp(np.linspace(0, nlen, nbin + 1), np.arange(nlen), np.sort(x))

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
                directory = validate_dir('figures/data_models')
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
            min_m=.4,
            max_m=4.5,
            min_mf=.4,
            max_mf=1.,
            min_beta=8e-3,
            max_beta=3.5e-2,
            val_frac=.2,
            random_state=None,
            num_param_values=100
    ):
        if len(rmodel.models) != len(bio_data):
            raise ValueError('The number of region models must be equal to the number of biological data points.')
        self.rmodel = rmodel
        self.map_bio = None
        m_shape = np.asarray(list(self.rmodel.get_model_parameter('m')))
        max_frac = np.asarray(list(self.rmodel.get_model_parameter('max_frac')))
        beta = np.asarray(list(self.rmodel.get_model_parameter('beta')))

        m_shape, self.m_mean, self.m_std = ParameterMap.normalise(m_shape)
        max_frac, self.max_frac_mean, self.max_frac_std = ParameterMap.normalise(max_frac)
        beta, self.beta_mean, self.beta_std = ParameterMap.normalise(beta)
        all_data_params = np.asarray([m_shape, max_frac, beta]).T

        self.data_params, self.data_params_val, self.bio_data, self.bio_data_val = train_test_split(
            all_data_params,
            bio_data,
            test_size=val_frac,
            random_state=random_state
        )

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
    def error(time_sample, real_data, est_mean, est_std):
        time_sample = np.asarray(time_sample)
        mask = time_sample > 0
        s = np.abs(np.asarray(est_mean) - np.asarray(real_data))
        # Discount for prediction within std
        s[
            np.logical_and(s <= est_std[time_sample], mask)
        ] *= s[np.logical_and(s <= est_std[time_sample], mask)] / est_std[
            np.logical_and(s <= est_std[time_sample], mask)]

        # Penalty for large std
        s += est_std * s

        return 100 * np.mean(s)

    def learn(self, num_cpus=1):
        pass

    def estimate(self, new_bio, time_scale=140):
        pass


class BayesianParameterMap(ParameterMap):
    def __init__(
            self,
            rmodel,
            bio_data,
            kernel_func_type,
            kernel_search_step=5e-4,
            kernel_search_thresh=1e-3,
            kernel_search_verbosity=0,
            kernel_param_min=.001,
            kernel_scaling_init=5.,
            noise_scaling_init=2.,
            min_m=.4,
            max_m=4.5,
            min_mf=.4,
            max_mf=1.,
            min_beta=8e-3,
            max_beta=3.5e-2,
            num_param_values=100,
            val_frac=.2,
            random_state=None,
            do_estimate_hp=True
    ):
        if len(rmodel.models) == 1:
            raise ValueError('The region model must contain at least two JMAK models.')

        super().__init__(
            rmodel,
            bio_data,
            min_m=min_m,
            max_m=max_m,
            min_mf=min_mf,
            max_mf=max_mf,
            min_beta=min_beta,
            max_beta=max_beta,
            num_param_values=num_param_values,
            val_frac=val_frac,
            random_state=random_state
        )
        # self.noise = noise
        if kernel_func_type == 'eqk':
            kernel_func = exponential_quadratic_kernel
            if do_estimate_hp:
                kernel_param, self.noise = self._optimise_eqk_param(
                    kernel_search_step,
                    kernel_search_thresh,
                    kernel_scaling_init,
                    noise_scaling_init,
                    kernel_param_min,
                    kernel_search_verbosity
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
        self.learnt_var = None

    @staticmethod
    def _calc_c(data, kernel, kernel_param, noise):
        C = np.asarray([
            kernel(x_i, x_j, *kernel_param)
            for x_i in data for x_j in data
        ]).reshape(data.shape[0], data.shape[0])
        C += np.eye(C.shape[0]) * noise
        inv_C = np.linalg.inv(C)
        return C, inv_C

    def jac(self, sub_jac, inv_C):
        return (
                -.5 * np.trace(inv_C.dot(sub_jac))
                + 0.5 * self.bio_data.dot(inv_C.dot(sub_jac.dot(self.bio_data.T)))
        )

    @staticmethod
    def _subjac_precision(size, precision):
        return - np.eye(size) * precision**(-2)

    def _optimise_eqk_param(self, step_size, thresh, scaling_init, noise_scaling_init, min_param, verbosity):
        theta_old = np.ones(4) * 999
        s = np.ones(4) * scaling_init
        s[-1] = noise_scaling_init
        theta_new = np.random.random(4) + s
        while np.linalg.norm(theta_old - theta_new) > thresh:
            theta_old = theta_new.copy()

            C_theta, inv_C_theta = self._calc_c(
                self.data_params,
                exponential_quadratic_kernel,
                theta_old[:3],
                1. / theta_old[3]
            )
            sub_jac_theta1 = np.asarray(
                [jac_eqk_theta_1(x_i, x_j, theta_old[1])
                 for x_i in self.data_params for x_j in self.data_params]
            ).reshape(self.data_params.shape[0], self.data_params.shape[0])

            sub_jac_theta2 = np.asarray(
                [jac_eqk_theta_2(x_i, x_j, *theta_old[:2])
                 for x_i in self.data_params for x_j in self.data_params]
            ).reshape(self.data_params.shape[0], self.data_params.shape[0])

            sub_jac_theta3 = np.asarray(
                [jac_eqk_theta_3(x_i, x_j) for x_i in self.data_params for x_j in self.data_params]
            ).reshape(self.data_params.shape[0], self.data_params.shape[0])

            sub_jac_precision = self._subjac_precision(self.data_params.shape[0], theta_old[3])
            jact1 = self.jac(sub_jac_theta1, inv_C_theta)
            jact2 = self.jac(sub_jac_theta2, inv_C_theta)
            jact3 = self.jac(sub_jac_theta3, inv_C_theta)
            jacp = self.jac(sub_jac_precision, inv_C_theta)
            jac_theta = np.asarray([jact1, jact2, jact3, jacp])
            theta_new = np.maximum(theta_old + step_size * jac_theta, min_param)

            if verbosity > 0:
                print('Diff %s' % np.linalg.norm(theta_old - theta_new))
                print('New Parameters %s' % theta_new)

        return theta_new[:3], 1. / theta_new[3]

    def _optimise_gaussian_param(self, step_size, thresh, scaling_init, noise_scaling_init, min_param, verbosity):
        sigma_sq_old = np.ones(2) * 999
        s = np.ones(2) * scaling_init
        s[-1] = noise_scaling_init
        sigma_sq_new = np.random.randn(2) + s
        while np.linalg.norm(sigma_sq_old - sigma_sq_new) > thresh:
            sigma_sq_old = sigma_sq_new.copy()

            C_sigma, inv_C_sigma = self._calc_c(
                self.data_params,
                gaussian_kernel,
                sigma_sq_old[0],
                1. / sigma_sq_old[1]
            )

            sub_jac_sigma = np.asarray(
                [jac_gaussian_kernel(x_i, x_j, sigma_sq_old[0])
                 for x_i in self.data_params for x_j in self.data_params]
            ).reshape(self.data_params.shape[0], self.data_params.shape[0])
            sub_jac_precision = self._subjac_precision(self.data_params.shape[0], sigma_sq_old[1])
            jac_sigma = self.jac(sub_jac_sigma, inv_C_sigma)
            jac_precision = self.jac(sub_jac_precision, inv_C_sigma)
            jac_param = np.asarray([jac_sigma, jac_precision])
            sigma_sq_new = np.maximum(sigma_sq_old + step_size * jac_param, min_param)
            if verbosity > 0:
                print('Diff %s' % np.linalg.norm(sigma_sq_old - sigma_sq_new))
                print('New parameter %s' % sigma_sq_new)

        return sigma_sq_new[0], 1. / sigma_sq_new[1]

    def _set_gramm(self):
        self.C, self.inv_C = self._calc_c(self.data_params, self.kernel.func, self.kernel.params, self.noise)

    @staticmethod
    def det_parameter_mapping(xn, y_data, kernel, weighted_t, inv_C, noise):
        k = np.asarray([kernel(x_j, xn) for x_j in y_data])
        m = k.dot(weighted_t)
        v = kernel(xn, xn) + noise - k.dot(inv_C).dot(k.T)
        return m, v

    def learn(self, num_cpus=1, estimate_time=140):
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
            error = self.error(np.arange(estimate_time), best_estimation, np.mean(p, axis=0), np.var(p, axis=0))
            val_error.append(error)

    def estimate(self, new_bio, time_scale=140):
        est_m_list, est_max_frac_list, est_beta_list, prediction = [], [], [], []
        for nb in new_bio:
            distances = np.abs(self.map_bio - nb)
            idc = np.where(distances - np.sqrt(self.learnt_var) <= np.min(distances))[0]
            est_m, est_max_frac, est_beta = (
                self.map_m.reshape(-1)[idc],
                self.map_max_frac.reshape(-1)[idc],
                self.map_beta.reshape(-1)[idc]
            )

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

        return np.asarray(prediction), est_m_list, est_max_frac_list, est_beta_list



