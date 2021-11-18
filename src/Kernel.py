import numpy as np


def gaussian_kernel(x1, x2, sigma_sq):
    return np.exp(- np.linalg.norm(x1 - x2) / (2 * sigma_sq))


def jac_gaussian_kernel(x1, x2, sigma_sq):
    return np.linalg.norm(x1 - x2) / (2 * sigma_sq**2) * gaussian_kernel(x1, x2, sigma_sq)


def exponential_quadratic_kernel(x1, x2, theta1, theta2, theta3):
    return (
        theta1 * np.exp(-theta2 * .5 * np.linalg.norm(x1 - x2))
        + theta3 * np.dot(x1, x2)
    )


def jac_eqk_theta_1(x1, x2, theta2):
    return np.exp(-.5 * theta2 * np.linalg.norm(x1 - x2))


def jac_eqk_theta_2(x1, x2, theta1, theta2):
    return -.5 * theta1 * np.linalg.norm(x1 - x2) * np.exp(-.5 * theta2 * np.linalg.norm(x1 - x2))


def jac_eqk_theta_3(x1, x2):
    return np.dot(x1, x2)

