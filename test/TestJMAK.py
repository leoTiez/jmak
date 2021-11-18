import unittest
import numpy as np
from src.Model import JMAK


def create_data(time_points, m, beta, max_frac):
    return np.asarray([(1 - np.exp(-(beta * t) ** m)) * max_frac for t in time_points])


class TestJMAK(unittest.TestCase):
    def test_1_init(self):
        time_points = [10, 10, 40, 100, 100, 100]
        data_points = [0.1, 0.15, 0.6, 0.6, 0.7, 0.75]
        name = 'Test Model'
        model = JMAK(time_points, data_points, name=name)
        self.assertListEqual(time_points, model.time_points)
        self.assertListEqual(data_points, model.data_points)
        self.assertIsNone(model.m)
        self.assertIsNone(model.beta)
        self.assertIsNone(model.max_frac)
        self.assertEqual(model.name, name)

        m = 2.
        beta = 0.01,
        max_frac = .8
        model = JMAK(time_points, data_points, m=m, beta=beta, max_frac=max_frac, name=name)
        self.assertListEqual(time_points, model.time_points)
        self.assertListEqual(data_points, model.data_points)
        self.assertEqual(model.m, m)
        self.assertEqual(model.beta, beta)
        self.assertEqual(model.max_frac, max_frac)
        self.assertEqual(model.name, name)

    def test_repair_fraction(self):
        orig_beta = 1 / 50.
        orig_m = 2.
        orig_max_frac = .75
        time_points = [0, 20, 25, 60, 80, 100, 120]
        name = 'Test Model'

        data = create_data(time_points, m=orig_m, beta=orig_beta, max_frac=orig_max_frac)
        model = JMAK(time_points, None, m=orig_m, beta=orig_beta, max_frac=orig_max_frac, name=name)
        for t, d in zip(time_points, data):
            rf = model.repair_fraction(t)
            self.assertEqual(d, rf)

    def test_estimate_shape_scale(self):
        orig_beta = 1/50.
        orig_m = 2.
        orig_max_frac = .75
        time_points = [1, 20, 25, 60, 80, 100, 120]  # Important no negative or zero values
        name = 'Test Model'
        data = create_data(time_points, m=orig_m, beta=orig_beta, max_frac=orig_max_frac)

        # Perfect estimation
        model = JMAK(time_points, data, name=name)
        pred_m, pred_beta, _ = model._estimate_shape_scale(orig_max_frac)
        self.assertAlmostEqual(pred_m, orig_m, 4)
        self.assertAlmostEqual(pred_beta, orig_beta, 6)

        # Difficult estimation
        time_points = [20, 60, 120]
        m_sim_thresh = 0.2
        beta_sim_thresh = 0.02
        data_noise = create_data(time_points, m=orig_m, beta=orig_beta, max_frac=orig_max_frac)
        data_noise += np.asarray([0.00683738, -0.00295627,  0.00194353])

        model = JMAK(time_points, data_noise, name=name)
        # Increase orig max frac to account for noise and to avoid runtime error
        est_max_frac = .76
        pred_m, pred_beta, _ = model._estimate_shape_scale(est_max_frac)
        self.assertTrue(np.abs(pred_m - orig_m) < m_sim_thresh, 'Difference is %.3f' % np.abs(pred_m - orig_m))
        self.assertTrue(
            np.abs(pred_beta - orig_beta) < beta_sim_thresh,
            'Difference is %.4f' % np.abs(pred_beta - orig_beta)
        )

    def test_estimate_parameters(self):
        orig_beta = 1 / 50.
        orig_m = 2.
        orig_max_frac = .75
        time_points = [20, 60, 120]
        m_sim_thresh = 0.2
        max_frac_thresh = 0.02
        beta_sim_thresh = 0.02
        name = 'Test Model'

        min_f = .5
        max_f = 1.
        delta_f = .01
        verbosity = 0

        data_noise = create_data(time_points, m=orig_m, beta=orig_beta, max_frac=orig_max_frac)
        data_noise += np.asarray([0.00683738, -0.00295627, 0.00194353])
        model = JMAK(time_points, data_noise, name=name)
        model.estimate_parameters(
            min_f=min_f,
            max_f=max_f,
            delta_f=delta_f,
            verbosity=verbosity
        )

        self.assertTrue(np.abs(model.m - orig_m) < m_sim_thresh, 'Difference is %.3f' % np.abs(model.m - orig_m))
        self.assertTrue(
            np.abs(model.beta - orig_beta) < beta_sim_thresh,
            'Difference is %.4f' % np.abs(model.beta - orig_beta)
        )
        self.assertTrue(
            np.abs(model.max_frac - orig_max_frac) < max_frac_thresh,
            'Difference is %.2f' % np.abs(model.max_frac - orig_max_frac)
        )


if __name__ == '__main__':
    unittest.main()
