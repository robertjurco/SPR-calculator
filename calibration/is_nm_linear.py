import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import scipy.optimize as optimize

################################### CONSTANTS ####################################
c = 299792458
hbar = 6.5822e-16


################################## GOLD DATA ####################################

data_AU = np.loadtxt('../data/gold_yakubovsky.txt')
wavelength_AU, n_AU_real, n_AU_imag = data_AU.T

spline_AU_real = CubicSpline(wavelength_AU, n_AU_real)
spline_AU_imag = CubicSpline(wavelength_AU, n_AU_imag)


def n_AU(wavelength):
    wavelength /= 1000
    return spline_AU_real(wavelength) - spline_AU_imag(wavelength) * 1j


################################## Ti DATA ####################################

data_Ti = np.loadtxt('../data/Ti.txt')
wavelength_Ti, n_Ti_real, n_Ti_imag = data_Ti.T

spline_Ti_real = CubicSpline(wavelength_Ti, n_Ti_real)
spline_Ti_imag = CubicSpline(wavelength_Ti, n_Ti_imag)


def n_Ti(wavelength):
    wavelength /= 1000
    return spline_Ti_real(wavelength) - spline_Ti_imag(wavelength) * 1j


################################## Calculator ####################################

def sqrte(z):
    if np.real(z) >= 0:
        return np.sqrt(z)
    else:
        return np.conj(np.sqrt(z)) # this definition is necessary to produce exponentially-decaying evanescent waves
def gamma(n, L, angle, polarity):
    N = len(n) - 2  # number of slabs
    angle = angle * np.pi / 180.0

    costh = np.zeros(len(n), complex)
    for i in range(len(n)):
        costh[i] = 1 - (n[0] * np.sin(angle) / n[i]) ** 2
        if np.real(costh[i]) >= 0:
            costh[i] = np.sqrt(costh[i])
        else:
            costh[i] = np.conj(np.sqrt(costh[i]))

    if polarity == 'TE':
        nT = n * costh  # transverse refractive indices
    else:
        nT = n / costh  # TM case, fails at 90 deg for left medium

    if N > 0:
        for i in range(N):
            L[i] *= costh[i+1]  # n(i) * l(i) * cos(th(i))

    r = np.zeros(N + 1, dtype=complex)
    for i in range(N + 1):
        r[i] = (nT[i] - nT[i + 1]) / (nT[i] + nT[i + 1])  # r(i) = (n(i - 1) - n(i)) / (n(i - 1) + n(i))

    Gamma = r[N]  #initialize Gamma at right-most interface

    for i in range(N - 1, -1, -1):
        delta = 2 * np.pi * L[i]  # phase thickness in i-th layer
        z = np.exp(-2j * delta)
        Gamma = (r[i] + Gamma * z) / (1 + r[i] * Gamma * z)

    return Gamma

##################################### Plasmon #########################################


def plasmon_n(x):
    wavelength, angle, thickness_au, thickness_ti, n = x
    n_gold = n_AU(wavelength)
    n_ti = n_Ti(wavelength)

    n = [1.51, n_ti, n_gold, n]
    L = [thickness_ti * n_ti / wavelength, thickness_au * n_gold / wavelength]

    gamma_return = gamma(n, L, angle, 'tm')
    return np.abs(gamma(n, L, angle, 'tm') ** 2)

def plasmon_minimum(x):
    bnds = ((650, 900), (x[1], x[1]), (x[2], x[2]), (x[3], x[3]), (x[4], x[4]))
    res = optimize.minimize(plasmon_n, x, bounds=bnds, tol=1e-10)
    return res.x[0]

##################################### Calibration curve #########################################

data_Calib = np.loadtxt('calibration_curve.txt')
n_calib, wavelength_calib = data_Calib.T

##################################### main code #########################################


# we optimize angle
xx = [750, 67, 50, 1.5, 1.33]
bnds = ((750, 750), (62, 68), (50, 50), (1.5, 1.5), (1.33, 1.33))
res = optimize.minimize(plasmon_n, xx, bounds=bnds, tol=1e-10)

# get data
n = np.linspace(1.33, 1.345, 100)
wavelength = np.zeros(100)

for i in range(100):
    # starting variables
    initial = [750, res.x[1], 50, 1.5, n[i]]
    wavelength[i] = plasmon_minimum(initial)

plt.subplots()

plt.ylabel('Sensor response - Wavelength [nm]')
plt.xlabel('Refractive index [RIU]')
plt.plot(n, wavelength, color='r', label='Theory')
plt.plot(n_calib, wavelength_calib, color='b', label='Measurement')
plt.grid()


plt.show()