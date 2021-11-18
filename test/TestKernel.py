import unittest
import numpy as np

import src.Kernel as kernel


class TestKernel(unittest.TestCase):
    def test_gaussian_kernel(self):
        # 1 dim values
        sigma_sq = 2
        x1 = 1.
        x2 = 2.
        exp_res = 0.778800783
        self.assertAlmostEqual(kernel.gaussian_kernel(x1, x2, sigma_sq=sigma_sq), exp_res, 8)

        x1 = 4.
        x2 = 1.
        exp_res = 0.472366553
        self.assertAlmostEqual(kernel.gaussian_kernel(x1, x2, sigma_sq=sigma_sq), exp_res, 8)

        sigma_sq = 9
        x1 = 1.
        x2 = 2.
        exp_res = 0.945959469
        self.assertAlmostEqual(kernel.gaussian_kernel(x1, x2, sigma_sq=sigma_sq), exp_res, 8)

        x1 = 4.
        x2 = 1.
        exp_res = 0.846481725
        self.assertAlmostEqual(kernel.gaussian_kernel(x1, x2, sigma_sq=sigma_sq), exp_res, 8)

        # 2 (n) dim values
        sigma_sq = 2
        x1 = np.asarray([1., 2.])
        x2 = np.asarray([5., 1])
        exp_res = 0.356729886
        self.assertAlmostEqual(kernel.gaussian_kernel(x1, x2, sigma_sq=sigma_sq), exp_res, 7)

        x1 = np.asarray([6., 7., 8.])
        x2 = np.asarray([1., 1., 1.])
        exp_res = 0.072655795
        self.assertAlmostEqual(kernel.gaussian_kernel(x1, x2, sigma_sq=sigma_sq), exp_res, 7)

        sigma_sq = 9
        x1 = np.asarray([1., 2.])
        x2 = np.asarray([5., 1])
        exp_res = 0.795279683
        self.assertAlmostEqual(kernel.gaussian_kernel(x1, x2, sigma_sq=sigma_sq), exp_res, 7)

        x1 = np.asarray([6., 7., 8.])
        x2 = np.asarray([1., 1., 1.])
        exp_res = 0.558404548
        self.assertAlmostEqual(kernel.gaussian_kernel(x1, x2, sigma_sq=sigma_sq), exp_res, 7)

    def test_exponential_quadratic_kernel(self):
        # 1 dim
        theta = np.asarray([2, 3, 4])
        x1 = 1.
        x2 = 2.
        exp_res = 8.44626032
        self.assertAlmostEqual(kernel.exponential_quadratic_kernel(x1, x2, *theta), exp_res, 8)

        x1 = 4.
        x2 = 1.
        exp_res = 16.022217993
        self.assertAlmostEqual(kernel.exponential_quadratic_kernel(x1, x2, *theta), exp_res, 8)

        theta = np.asarray([1, 1, 1])
        x1 = 1.
        x2 = 2.
        exp_res = 2.60653066
        self.assertAlmostEqual(kernel.exponential_quadratic_kernel(x1, x2, *theta), exp_res, 8)

        x1 = 4.
        x2 = 1.
        exp_res = 4.22313016
        self.assertAlmostEqual(kernel.exponential_quadratic_kernel(x1, x2, *theta), exp_res, 8)

        # 2 (n) dim
        theta = np.asarray([2, 3, 4])
        x1 = np.asarray([1., 2.])
        x2 = np.asarray([5., 1])
        exp_res = 28.004121611
        self.assertAlmostEqual(kernel.exponential_quadratic_kernel(x1, x2, *theta), exp_res, 7)

        x1 = np.asarray([6., 7., 8.])
        x2 = np.asarray([1., 1., 1.])
        exp_res = 84.000000294
        self.assertAlmostEqual(kernel.exponential_quadratic_kernel(x1, x2, *theta), exp_res, 7)

        theta = np.asarray([1, 1, 1])
        x1 = np.asarray([1., 2.])
        x2 = np.asarray([5., 1])
        exp_res = 7.127256211
        self.assertAlmostEqual(kernel.exponential_quadratic_kernel(x1, x2, *theta), exp_res, 7)

        x1 = np.asarray([6., 7., 8.])
        x2 = np.asarray([1., 1., 1.])
        exp_res = 21.005278865
        self.assertAlmostEqual(kernel.exponential_quadratic_kernel(x1, x2, *theta), exp_res, 7)


if __name__ == '__main__':
    unittest.main()
