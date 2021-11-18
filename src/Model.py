import numpy as np
import pandas as pd
import statsmodels.api as smapi
import matplotlib.pyplot as plt
import matplotlib.colors as cls
from abc import ABC
import multiprocessing
import warnings

from src.Utils import validate_dir
from src.Kernel import *


class JMAK:
    def __init__(self, time_points, data_points, m=None, max_frac=None, beta=None, name=''):
        self.time_points = time_points.reshape(-1)
        self.data_points = data_points.reshape(-1)
        self.m = m
        self.max_frac = max_frac
        self.beta = beta
        self.name = name

    def repair_fraction(self, time):
        if self.m is None or self.max_frac is None or self.beta is None:
            raise ValueError(
                'Model parameters have not been set.\n'
                'Run estimate_parameters to fit the model to the data, or set the parameters manually'
            )

        return (1 - np.exp(- np.power(self.beta * time, self.m))) * self.max_frac

    def _estimate_shape_scale(self, max_frac):
        lin_est = smapi.OLS(
            np.log(np.log(1. / (1 - (np.asarray(self.data_points) - np.finfo('float').eps) / max_frac))),
            smapi.add_constant(np.log(self.time_points))
        )
        result = lin_est.fit()
        return (
            result.params[1],  # shape
            np.exp(result.params[0] / result.params[1]),  # scale
            result.fvalue  # fvalue
        )

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

        fval = []
        beta_est = []
        m_est = []
        if verbosity > 1:
            fig = plt.figure(figsize=figsize)
            num_cols = int(np.sqrt((max_f - min_f) / delta_f))
            num_rows = int(np.ceil(((max_f - min_f) / delta_f) / num_cols))
        for num, mf in enumerate(np.arange(min_f, max_f + delta_f, delta_f)):
            if verbosity > 0:
                print('Estimate parameters for maximum fraction set to %.2f' % mf)
            m, beta, f = self._estimate_shape_scale(mf)
            if verbosity > 0:
                print('Estimated parameters for maximum fraction %.2f are\nm=\t%.3f\nbeta=\t%.5f' % (mf, m, beta))
                if verbosity > 1:
                    ax = fig.add_subplot(num_rows, num_cols, num)
                    self._plot_in_logspace(ax, m=m, beta=beta, max_frac=mf)

            fval.append(f)
            beta_est.append(beta)
            m_est.append(m)

        fval = np.asarray(fval)
        if np.all(np.isnan(fval)):
            self.m = None
            self.beta = None
            self.max_frac = None
            return

        idx = np.argmax(fval[~np.isnan(fval)])
        self.beta = np.asarray(beta_est)[~np.isnan(fval)][idx]
        self.m = np.asarray(m_est)[~np.isnan(fval)][idx]
        self.max_frac = np.arange(min_f, max_f + delta_f, delta_f)[~np.isnan(fval)][idx]

        if verbosity > 1:
            if save_fig:
                directory = validate_dir('figures/data_models')
                fig.savefig('%s/%s_model_fitting.png' % (directory, save_prefix))
                plt.close('all')
            else:
                plt.show()

    def _plot_in_logspace(self, ax, m, beta, max_frac):
        ax.scatter(
            np.log(self.time_points),
            np.log(np.log(1. / (1 - (np.asarray(self.data_points) - np.finfo('float').eps) / max_frac))),
            label='Data'
        )
        scale = np.linspace(0, np.max(self.time_points), 5)
        ax.plot(
            np.log(scale),
            [m * tlog + m * np.log(beta) for tlog in np.log(scale)],
            label='Estimation'
        )
        ax.set_title('Log space')
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
    def iteration(i, time, data, n, min_f, max_f, delta_f, verbosity, figsize, save_fig, save_prefix):
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
        if num_cpus <= 0:
            for num, (t, d) in enumerate(zip(self.time_points, self.data_points)):
                self.models.append(self.iteration(
                    num, t, d, names[num] if names is not None else '',
                    min_f, max_f, delta_f, verbosity-1, figsize, save_fig, save_prefix))
        else:
            if verbosity > 0:
                warnings.warn('Verbosity is larger than 0 but multiprocessing is used. Ignore verbosity flag.')
            num_cpus = np.minimum(multiprocessing.cpu_count() - 1, num_cpus)
            with multiprocessing.Pool(processes=num_cpus) as parallel:
                models = []
                for num, (t, d) in enumerate(zip(self.time_points, self.data_points)):
                    models.append(
                        parallel.apply_async(self.iteration, args=(num, t, d, names[num] if names is not None else '',
                                                              min_f, max_f, delta_f, 0, figsize,
                                                              save_fig, save_prefix,))
                    )
                parallel.close()
                parallel.join()
                self.models = [ar.get() for ar in models]

    def get_model_parameter(self, identifier='m'):
        if identifier == 'm':
            return filter(lambda y: y is not None, map(lambda x: x.m, self.models))
        elif identifier == 'beta':
            return filter(lambda y: y is not None, map(lambda x: x.beta, self.models))
        elif identifier == 'max_frac':
            return filter(lambda y: y is not None, map(lambda x: x.max_frac, self.models))
        elif identifier == 'name':
            return filter(lambda y: y is not None, map(lambda x: x.name if x.m is not None else None, self.models))
        else:
            raise ValueError('Invalid identifier')

    def load_models(self, file_path=''):
        df = pd.read_csv(file_path)
        for entry, t, d in zip(df.iterrows(), self.time_points, self.data_points):
            entry = entry[1]
            model = JMAK(t, d, m=entry['m'], max_frac=entry['max_frac'], beta=entry['beta'], name=entry['name'])
            self.models.append(model)

    def to_file(self, file_name=''):
        names = list(self.get_model_parameter('name'))
        m_list = list(self.get_model_parameter('m'))
        max_frac_list = self.get_model_parameter('max_frac')
        beta_list = self.get_model_parameter('beta')

        if not names:
            names = [''] * len(m_list)
        df = pd.DataFrame(zip(names, m_list, max_frac_list, beta_list), columns=['name', 'm', 'max_frac', 'beta'])
        directory = validate_dir('data/jmak')
        df.to_csv('%s/%s.csv' % (directory, file_name), sep='\t')

    def plot_parameter_histogram(
            self,
            color,
            fcolor='white',
            bins=(30, 30),
            norm_gamma=.3,
            figsize=(4, 10),
            save_fig=True,
            save_prefix=''
    ):
        def plot_hist(param_x, param_y, a):
            hist, _, _ = np.histogram2d(param_y, param_x, bins=bins)
            a.imshow(hist, norm=cls.PowerNorm(gamma=norm_gamma), cmap=color, origin='lower')
            for i in range(hist.shape[0]):
                for j in range(hist.shape[1]):
                    if hist[i, j] == 0:
                        continue
                    a.text(j, i, '%i' % hist[i, j], ha='center', va='center', color=fcolor, fontsize=8)

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

        params_m = list(self.get_model_parameter('m'))
        params_beta = list(self.get_model_parameter('beta'))
        params_max_frac = list(self.get_model_parameter('max_frac'))
        fig, ax = plt.subplots(3, 1, figsize=figsize)

        plot_hist(params_m, params_beta, ax[0])
        plot_hist(params_m, params_max_frac, ax[1])
        plot_hist(params_max_frac, params_beta, ax[2])

        ax[0].set_xlabel('m')
        ax[0].set_ylabel(r'$\beta$')

        ax[1].set_xlabel('m')
        ax[1].set_ylabel('Maximum fraction')

        ax[2].set_xlabel('Maximum fraction')
        ax[2].set_ylabel(r'$\beta$')

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
            norm=cls.PowerNorm(1),
            save_fig=True,
            save_prefix=False
    ):
        plt.figure(figsize=figsize)
        scatter = plt.scatter(
            list(self.get_model_parameter('m')),
            list(self.get_model_parameter('max_frac')),
            c=cgradient,
            s=(np.asarray(list(self.get_model_parameter('beta'))) * size_scaling)**size_power,
            cmap=cmap,
            norm=norm,
            alpha=alpha
        )
        plt.xlabel('m')
        plt.ylabel('Maximum fraction')
        plt.title('Parameter distribution %s' % self.name)
        plt.colorbar(scatter)
        handles, labels = scatter.legend_elements(
            'sizes',
            num=num_handles,
            func=lambda x: np.power(x, 1./size_power) / size_scaling
        )
        plt.legend(handles, labels, handletextpad=2, frameon=False)

        if save_fig:
            directory = validate_dir('figures/data_models')
            plt.savefig('%s/%s_%s_model_parameter.png' % (directory, save_prefix, self.name))
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
            num_param_values=100
    ):
        self.rmodel = rmodel
        self.bio_data = bio_data
        self.map_bio = None
        m_shape = np.asarray(list(self.rmodel.get_model_parameter('m')))
        max_frac = np.asarray(list(self.rmodel.get_model_parameter('max_frac')))
        beta = np.asarray(list(self.rmodel.get_model_parameter('beta')))

        m_shape, self.m_mean, self.m_std = ParameterMap.normalise(m_shape)
        max_frac, self.max_frac_mean, self.max_frac_std = ParameterMap.normalise(max_frac)
        beta, self.beta_mean, self.beta_std = ParameterMap.normalise(beta)
        self.data_params = np.asarray([m_shape, max_frac, beta]).T

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
        s = np.abs(np.asarray(est_mean)[time_sample] - np.asarray(real_data))
        s[s <= est_std[time_sample]] *= est_std[time_sample][s <= est_std[time_sample]] / .5

        mean_abs_error = np.sum(s) / float(len(s))
        mean_val = np.sum(np.asarray(real_data)) / float(len(real_data))
        return 100. * mean_abs_error / mean_val

    def learn(self):
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
            do_estimate_hp=True
    ):
        if len(rmodel.models) == 1:
            raise ValueError('The region model must contain at least to JMAK models.')

        super().__init__(
            rmodel,
            bio_data,
            min_m=min_m,
            max_m=max_m,
            min_mf=min_mf,
            max_mf=max_mf,
            min_beta=min_beta,
            max_beta=max_beta,
            num_param_values=num_param_values
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
    def _subjac_precision(size, noise):
        return - np.eye(size) * noise**(-2)

    def _optimise_eqk_param(self, step_size, thresh, scaling_init, noise_scaling_init, min_param, verbosity):
        theta_old = np.ones(4) * 999
        s = np.ones(4) * scaling_init
        s[-1] = noise_scaling_init
        theta_new = np.random.randn(4) + s
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

    def learn(self):
        self._set_gramm()

        mean, var = [], []
        weighted_t = self.inv_C.dot(self.bio_data)
        for xn in self.map_param:
            k = np.asarray([self.kernel(x_j, xn) for x_j in self.data_params])
            m = k.dot(weighted_t)
            v = self.kernel(xn, xn) + self.noise - k.dot(self.inv_C).dot(k.T)
            mean.append(m)
            var.append(v)

        self.map_bio = np.asarray(mean)
        self.learnt_var = np.asarray(var)

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
            prediction.append([tp_model.repair_fraction(t) for t in range(time_scale)])

        return np.asarray(prediction), est_m_list, est_max_frac_list, est_beta_list



