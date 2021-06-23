import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.shape_base import expand_dims
import scipy
import scipy.integrate
from tqdm.notebook import tqdm
import crank_nicolson_numba.generic as cn
import itertools
# For parallelization
from joblib import Parallel, delayed


def D(I, a, c=1.0, halved=False):
    return c * np.power(I, a) * (0.5 if halved else 1.0)


def standard_c(I_min, I_max, a):
    result = scipy.integrate.quad(lambda x: D(x, a), I_min, I_max)[0]
    if result == 0.0:
        return np.inf
    else:
        return 1/result


def afpt(I_min, I_max, a, c=None):
    if c is None:
        c = standard_c(0.0, I_max, a)
    return scipy.integrate.quad(
        lambda x: 2*x/D(x, a, c=c),
        I_min,
        I_max
    )[0]


def single_x(I, I_max, a, c):
    return -scipy.integrate.quad(lambda x: 1/np.sqrt(D(x, a, c)), I, I_max)[0]


x = np.vectorize(single_x, excluded=["a", "c"])


def nu(I, a, c):
    return (np.sqrt(c) * (a * 0.5)) * np.power(I, (a * 0.5) - 1)


def current_peak_time(I_0, I_max, a, c):
    return 2 * (
        np.sqrt(nu(I_0, a, c)**2 * x(I_0, I_max, a, c)**2 + 9) - 3) / nu(I_0, a, c)**2


def current_peak_value(I_0, I_max, a, c):
    return -x(I_0, I_max, a, c)*np.exp(-nu(I_0, a, c)**2*(x(I_0, I_max, a, c) + (np.sqrt(nu(I_0, a, c)**2*x(I_0, I_max, a, c)**2 + 9) - 3)/nu(I_0, a, c))**2/(4*(np.sqrt(nu(I_0, a, c)**2*x(I_0, I_max, a, c)**2 + 9) - 3)))/(4*np.sqrt(np.pi)*((np.sqrt(nu(I_0, a, c)**2*x(I_0, I_max, a, c)**2 + 9) - 3)/nu(I_0, a, c)**2)**(3/2))


def current_derivative(t, I, I_max, a, c):
    return -np.sqrt(2)*x(I, I_max, a, c)*(-nu(I, a, c)*(nu(I, a, c)*t/2 + x(I, I_max, a, c))/(2*t) + (nu(I, a, c)*t/2 + x(I, I_max, a, c))**2/(2*t**2))*np.exp(-(nu(I, a, c)*t/2 + x(I, I_max, a, c))**2/(2*t))/(2*np.sqrt(np.pi)*t**(3/2)) + 3*np.sqrt(2)*x(I, I_max, a, c)*np.exp(-(nu(I, a, c)*t/2 + x(I, I_max, a, c))**2/(2*t))/(4*np.sqrt(np.pi)*t**(5/2))


def current_second_derivative(t, I, I_max, a, c):
    return np.sqrt(2)*x(I, I_max, a, c)*(-nu(I, a, c)**4*t**4 - 24*nu(I, a, c)**2*t**3 + 8*nu(I, a, c)**2*t**2*x(I, I_max, a, c)**2 - 240*t**2 + 160*t*x(I, I_max, a, c)**2 - 16*x(I, I_max, a, c)**4)*np.exp(-nu(I, a, c)**2*t/8 - nu(I, a, c)*x(I, I_max, a, c)/2 - x(I, I_max, a, c)**2/(2*t))/(128*np.sqrt(np.pi)*t**(11/2))


def current_point(t, I, I_max, a, c):
    if t == 0:
        return 0.0
    return -x(I, I_max, a, c) / (t * np.sqrt(2*np.pi*t)) * np.exp(-(x(I, I_max, a, c)+((nu(I, a, c)/2)*t))**2/(2*t))


def current_uniform(t, I, I_max, I_int_min, a, c):
    return [scipy.integrate.quad(lambda x: current_point(a_t, x, I_max, a, c), I_int_min, I)[0] for a_t in t]


def current_generic(t, rho, I_max, I_int_min, a, c):
    return [
        scipy.integrate.quad(
            lambda x:
                current_point(a_t, x, I_max, a, c) * rho(x),
            I_int_min, I_max
        )[0] for a_t in t
    ]


def compute_current_with_peak(I_0, I_max, a, t_sampling=1000, I_sampling=50000, I_min=0.0):
    c = 1/scipy.integrate.quad(lambda x: D(x, a), I_min, I_max)[0]
    I_linspace, dI = np.linspace(I_min, I_max, I_sampling, retstep=True)
    sigma = dI * 5

    def rho_0(I):
        return np.exp(-0.5 * (I - I_0)**2/sigma**2) / (sigma*np.sqrt(2*np.pi))

    ana_current_peak_time = current_peak_time(I_0, I_max, a, c)
    ana_current_peak_value = current_peak_value(
        I_0, I_max, a, c)

    dt = ana_current_peak_time/t_sampling
    engine = cn.cn_generic(I_min, I_max, rho_0(
        I_linspace), dt, lambda x: D(x, a, c, halved=True), normalize=False)
    times, current = engine.current(t_sampling*2, 1)

    num_current_peak = np.max(current)
    num_current_time = times[np.argmax(current)]

    return times, current, ana_current_peak_time, ana_current_peak_value, num_current_time, num_current_peak


def compute_generic_current_with_peak(I_0, I_max, rho, a, t_sampling=1000, t_multiplier=2, I_sampling=50000, I_min=0.0):
    c = 1/scipy.integrate.quad(lambda x: D(x, a), I_min, I_max)[0]
    I_linspace, dI = np.linspace(I_min, I_max, I_sampling, retstep=True)
    ana_current_peak_time = current_peak_time(I_0, I_max, a, c)

    dt = ana_current_peak_time/t_sampling
    engine = cn.cn_generic(
        I_min, I_max,
        rho(I_linspace),
        dt,
        lambda x: D(x, a, c, halved=True),
        normalize=False
    )
    times, current = engine.current(t_sampling * t_multiplier, 1)

    num_current_peak = np.max(current)
    num_current_time = times[np.argmax(current)]

    return c, times, current, ana_current_peak_time, num_current_time, num_current_peak


def locate_generic_maximum(I_0, I_max, rho, a, starting_point=None, I_min_int=0.1):
    c = standard_c(0.0, I_max, a)
    if starting_point is None:
        starting_point = current_peak_time(I_0, I_max, a, c)
    result = scipy.optimize.fmin(
        lambda x: - current_generic(x, rho, I_max, I_min_int, a, c)[0],
        starting_point,
        disp=False
    )
    return result


def analytical_recover(I_0, I_max, a, ratio=0.1, I_min=0.0):
    c = standard_c(I_min, I_max, a)
    t0 = current_peak_time(I_0, I_max, a, c)
    m0 = current_peak_value(I_0, I_max, a, c)
    result = scipy.optimize.fmin(
        lambda x: np.absolute(current_point(x, I_0, I_max, a, c) - m0 * ratio),
        t0*0.5,
        disp=False
    )
    return result[0]
