import unittest
import numpy as np

from src.Model import BayesianParameterMap, RegionModel
import src.Kernel as kernel


def create_model(do_estimate_hp=True):
    # Create and fit rmodel
    np.random.seed(0)
    time_points = np.asarray([[20, 60, 120], [20, 60, 120]])
    data_points = np.asarray([[.2, .6, .7], [.1, .5, .6]])
    rmodel = RegionModel(time_points=time_points, data_points=data_points, name='Test Model')
    rmodel.fit_models(min_f=.5, max_f=1, delta_f=.01)

    m = np.asarray(list(rmodel.get_model_parameter('m')))
    mf = np.asarray(list(rmodel.get_model_parameter('max_frac')))
    beta = np.asarray(list(rmodel.get_model_parameter('beta')))

    bio_data = np.asarray([10, 20])
    kernel_func_type = 'eqk'
    kernel_func = kernel.exponential_quadratic_kernel
    kernel_search_step = 0.0001
    kernel_search_thresh = .001
    kernel_search_verbosity = 1
    min_m = 1.
    max_m = 2.
    min_mf = .5
    max_mf = .9
    min_beta = 0.01
    max_beta = 0.03
    num_param_values = 2

    model = BayesianParameterMap(
        rmodel=rmodel,
        bio_data=bio_data,
        kernel_func_type=kernel_func_type,
        kernel_search_step=kernel_search_step,
        kernel_search_thresh=kernel_search_thresh,
        kernel_search_verbosity=kernel_search_verbosity,
        min_m=min_m,
        max_m=max_m,
        min_mf=min_mf,
        max_mf=max_mf,
        min_beta=min_beta,
        max_beta=max_beta,
        num_param_values=num_param_values,
        do_estimate_hp=do_estimate_hp
    )
    return model, (rmodel, m, mf, beta, bio_data,
                   kernel_func, min_m, max_m, min_mf, max_mf,
                   min_beta, max_beta, num_param_values)


class TestBayesianParameterMap(unittest.TestCase):
    def test_1_init(self):
        model, (rmodel, m, mf, beta, bio_data,
                kernel_func, min_m, max_m, min_mf, max_mf,
                min_beta, max_beta, num_param_values) = create_model()

        kernel_params_tolerance = .5
        noise_params_tolerance = .5
        exp_kernel_params = np.asarray([23.5, 5.3, 8.5])
        exp_noise = .33

        # Check only for memory address as the TestRegionModel has already tested the RegionModel
        self.assertEqual(model.rmodel, rmodel)
        np.testing.assert_equal(model.bio_data, bio_data)
        self.assertIsNone(model.map_bio)
        self.assertEqual(model.m_mean, np.mean(m))
        self.assertEqual(model.m_std, np.std(m))
        self.assertEqual(model.max_frac_mean, np.mean(mf))
        self.assertEqual(model.max_frac_std, np.std(mf))
        self.assertEqual(model.beta_mean, np.mean(beta))
        self.assertEqual(model.beta_std, np.std(beta))

        # Normalise m, max fraction, and beta to zero mean and std of 1
        m_norm = (m - np.mean(m)) / np.std(m)
        mf_norm = (mf - np.mean(mf)) / np.std(mf)
        beta_norm = (beta - np.mean(beta)) / np.std(beta)
        np.testing.assert_equal(model.data_params, np.asarray([m_norm, mf_norm, beta_norm]).T)

        # Testing for the parameter map
        m_map = np.linspace(min_m, max_m, num_param_values)
        mf_map = np.linspace(min_mf, max_mf, num_param_values)
        beta_map = np.linspace(min_beta, max_beta, num_param_values)
        m_map, mf_map, beta_map = np.meshgrid(m_map, mf_map, beta_map)
        np.testing.assert_equal(model.map_m, m_map)
        np.testing.assert_equal(model.map_max_frac, mf_map)
        np.testing.assert_equal(model.map_beta, beta_map)
        param_map = np.asarray([
            (m_map.reshape(-1) - np.mean(m)) / np.std(m),
            (mf_map.reshape(-1) - np.mean(mf)) / np.std(mf),
            (beta_map.reshape(-1) - np.mean(beta)) / np.std(beta)
        ]).T
        np.testing.assert_equal(model.map_param, param_map)

        self.assertEqual(model.kernel.func, kernel.exponential_quadratic_kernel)
        self.assertTrue(np.all(np.abs(model.kernel.params - exp_kernel_params) < kernel_params_tolerance))
        self.assertTrue(np.abs(model.noise - exp_noise) < noise_params_tolerance)
        self.assertIsNone(model.C)
        self.assertIsNone(model.inv_C)
        self.assertIsNone(model.learnt_var)

    def test_calc_c(self):
        model, (_, _, _, _, _,
                _, _, _, _,
                _, _, _, _) = create_model(do_estimate_hp=False)
        noise = .1
        data = np.asarray([[0, 0], [1, 1]])
        kernel_param = np.asarray([1, 1, 0])
        k = kernel.exponential_quadratic_kernel
        C_model, C_inv_model = model._calc_c(data, k, kernel_param, noise)
        C_exp = np.asarray([[1.1, 0.49306869], [0.49306869, 1.1]])
        C_inv_exp = np.asarray([[1.13767612, -0.5099568], [-0.5099568 ,1.13767612]])
        np.testing.assert_almost_equal(C_model, C_exp, 7)
        np.testing.assert_almost_equal(C_inv_model, C_inv_exp, 7)

    def test_estimate(self):
        m_tolerance = 0.2
        mf_tolerance = 0.051
        beta_tolerance = 0.02

        model, (rmodel, m, mf, beta, bio_data,
                kernel_func, min_m, max_m, min_mf, max_mf,
                min_beta, max_beta, num_param_values) = create_model()
        m_map = np.linspace(min_m, max_m, num_param_values)
        mf_map = np.linspace(min_mf, max_mf, num_param_values)
        beta_map = np.linspace(min_beta, max_beta, num_param_values)

        model.learn()

        # Correct bio data input should return the correct parameters
        _, est_m, est_max_frac, est_beta = model.estimate(bio_data, 140)
        for om, em, omf, emf, obeta, ebeta in zip(m, est_m, mf, est_max_frac, beta, est_beta):
            self.assertTrue(
                np.abs(om - np.mean(em)) < m_tolerance,
                'Actual difference is %.3f' % (np.abs(om - np.mean(em)))
            )
            self.assertTrue(
                np.abs(omf - np.mean(emf)) < mf_tolerance,
                'Actual difference is %.4f' % (np.abs(omf - np.mean(emf)))
            )
            self.assertTrue(
                np.abs(obeta - np.mean(ebeta)) < beta_tolerance,
                'Actual difference is %.4f' % (np.abs(obeta - np.mean(ebeta)))
            )


if __name__ == '__main__':
    unittest.main()
