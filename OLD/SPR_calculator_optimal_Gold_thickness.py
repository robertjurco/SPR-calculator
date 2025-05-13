import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import scipy.optimize as optimize

################################### CONSTANTS ####################################
c = 299792458
hbar = 6.5822e-16

################################ Platinum DATA ####################################

data_Ag = np.loadtxt('../data/Ag.txt')
wavelength_Ag, n_Ag_real, n_Ag_imag = data_Ag.T

spline_Ag_real = CubicSpline(wavelength_Ag, n_Ag_real)
spline_Ag_imag = CubicSpline(wavelength_Ag, n_Ag_imag)


def n_Ag(wavelength):
    wavelength /= 1000
    return spline_Ag_real(wavelength) - spline_Ag_imag(wavelength) * 1j


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
    wavelength, angle, thickness_au, thickness_ti = x

    thickness_layer = 0.0
    n_layer = 1.33

    n_gold = n_AU(wavelength)
    n_ti = n_Ti(wavelength)

    n = [1.51, n_ti, n_gold, n_layer, 1.33]
    L = [thickness_ti * n_ti / wavelength, thickness_au * n_gold / wavelength, thickness_layer * n_layer / wavelength]

    gamma_return = gamma(n, L, angle, 'tm')
    return np.abs(gamma(n, L, angle, 'tm') ** 2)

def plasmon_ndn(x):
    wavelength, angle, thickness_au, thickness_ti = x

    thickness_layer = 0.0
    n_layer = 1.40

    n_gold = n_AU(wavelength)
    n_ti = n_Ti(wavelength)

    n = [1.51, n_ti, n_gold, n_layer, 1.33005]
    L = [thickness_ti * n_ti / wavelength, thickness_au * n_gold / wavelength, thickness_layer * n_layer / wavelength]

    gamma_return = gamma(n, L, angle, 'tm')
    return np.abs(gamma(n, L, angle, 'tm') ** 2)

def optimum(x):
    wavelength, angle, thickness_au, thickness_ti = x
    return (plasmon_n(x) - plasmon_ndn(x)) / 0.00005


##################################### SPECTRAL #########################################

def minimum_n(x):
    wavelength, angle, thickness_au, thickness_ti = x

    thickness_layer = 0.0
    n_layer = 1.33

    n_gold = n_AU(wavelength)
    n_ti = n_Ti(wavelength)

    n = [1.51, n_ti, n_gold, n_layer, 1.33]
    L = [thickness_ti * n_ti / wavelength, thickness_au * n_gold / wavelength, thickness_layer * n_layer / wavelength]

    gamma_return = gamma(n, L, angle, 'tm')
    return np.abs(gamma(n, L, angle, 'tm') ** 2)

def minimum_ndn(x):
    wavelength, angle, thickness_au, thickness_ti = x

    thickness_layer = 0.0
    n_layer = 0.0

    n_gold = n_AU(wavelength)
    n_ti = n_Ti(wavelength)

    n = [1.51, n_ti, n_gold, n_layer, 1.3305]
    L = [thickness_ti * n_ti / wavelength, thickness_au * n_gold / wavelength, thickness_layer * n_layer / wavelength]

    gamma_return = gamma(n, L, angle, 'tm')
    return np.abs(gamma(n, L, angle, 'tm') ** 2)

def optimum_(x):
    wavelength, angle, thickness_au, thickness_ti = x

    res = optimize.minimize(optimum, initial, bounds=bnds, tol=1e-10)
    spectral_range = 300
    return (minimum_n(x) - minimum_ndn(x)) / 0.0005 / spectral_range

##################################### main code #########################################


initial = [70, 62, 43, 1.5]
bnds = ((1500, 1600), (62.0, 62.6), (40, 42), (1.5, 1.5))
res = optimize.minimize(optimum, initial, bounds=bnds, tol=1e-10)

print(res.x)

wavelength = np.linspace(1300, 1700, 1000)
result = np.zeros(1000)
result_n = np.zeros(1000)
result_ndn = np.zeros(1000)

for i in range(1000):
    result[i] = -1*optimum([wavelength[i], res.x[1], res.x[2], res.x[3]])
    result_n[i] = plasmon_n([wavelength[i], res.x[1], res.x[2], res.x[3]])
    result_ndn[i] = plasmon_ndn([wavelength[i], res.x[1], res.x[2], res.x[3]])

fig, ax1 = plt.subplots()

ax1.set_xlabel('Wavelength [nm]')
ax1.set_ylabel('Reflectivty [a.u.]')
ax1.plot(wavelength, result_n, color='r', label='n = 1.33')
ax1.plot(wavelength, result_ndn, color='b', label='n = 1.4')
plt.legend(loc='lower left')

# Adding Twin Axes

ax2 = ax1.twinx()
ax2.set_ylabel('Bulk sensitivity [1/RIU]')
ax2.plot(wavelength, result, color='k', label='Surface sensitivity')
ax2.grid()
plt.legend(loc='upper right')

plt.title("λ = " + str(round(res.x[0],2)) + "nm, ϴ = " + str(round(res.x[1],2)) + "°, Ti = "+ str(round(res.x[3],2)) + "nm followed by Gold = "+ str(round(res.x[2],2)) + "nm.")

plt.tight_layout()
plt.show()
