import unittest
import numpy as np

from src.Model import RegionModel, JMAK


def create_data(time_points, m, beta, max_frac):
    return np.asarray([(1 - np.exp(-(beta * t) ** m)) * max_frac for t in time_points])


def create_model():
    name = 'Test Model'
    time_points = np.tile(np.arange(1, 5) * 20, 4).reshape(4, 4)
    orig_beta = 1 / 50.
    orig_m = 2.
    orig_max_frac = .75
    data_points = np.tile(
        create_data(time_points[0], m=orig_m, beta=orig_beta, max_frac=orig_max_frac), 4).reshape(4, 4)
    data_points += np.asarray([
        [0.01385854, -0.00889024,  0.01075381, -0.00169284],
        [-0.00284679,  0.00527872, -0.01318768,  0.01464434],
        [ 0.0091277, -0.01448575,  0.00151347,  0.01828714],
        [-0.00794384, -0.00643361, -0.00163115,  0.00954813]])

    return (
        RegionModel(time_points=time_points, data_points=data_points, name=name),
        time_points,
        data_points,
        name,
        orig_m,
        orig_beta,
        orig_max_frac
    )


class TestRegionModel(unittest.TestCase):
    def test_1_init(self):
        model, time_points, data_points, name, _, _, _ = create_model()
        np.testing.assert_equal(model.data_points, data_points)
        np.testing.assert_equal(model.time_points, time_points)
        self.assertEqual(model.name, name)
        self.assertEqual(len(model.models), 0)

    def test_get_parameters(self):
        model, time_points, data_points, name, _, _, _ = create_model()
        jmak_model_list = [
            JMAK(time_points=time_points[0], data_points=data_points[0], m=2., beta=0.02, max_frac=0.75 + delta_f)
            for delta_f in np.arange(0.0, 0.04, 0.01)
        ]
        model.models = jmak_model_list

        self.assertListEqual(list(model.get_model_parameter('m')), [2., 2., 2., 2.])
        self.assertListEqual(list(model.get_model_parameter('beta')), [.02, .02, .02, .02])
        self.assertListEqual(list(model.get_model_parameter('max_frac')), [.75, .76, .77, .78])

    def test_fit_models(self):
        min_f = .5
        max_f = 1.
        delta_f = .01
        verbosity = 0
        m_sim_thresh = .2
        max_frac_sim_thresh = .1
        beta_sim_thresh = .025

        model, time_points, data_points, name, orig_m, orig_beta, orig_max_frac = create_model()
        model.fit_models(
            min_f=min_f,
            max_f=max_f,
            delta_f=delta_f,
            verbosity=verbosity
        )

        for m, beta, mf in zip(
                model.get_model_parameter('m'),
                model.get_model_parameter('beta'),
                model.get_model_parameter('max_frac'),
        ):
            self.assertTrue(np.abs(orig_m - m) < m_sim_thresh)
            self.assertTrue(np.abs(orig_beta - beta) < beta_sim_thresh)
            self.assertTrue(np.abs(orig_max_frac - mf) < max_frac_sim_thresh)


if __name__ == '__main__':
    unittest.main()
