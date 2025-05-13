import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

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

################################## Ag DATA ####################################

data_AG = np.loadtxt('../data/Ag.txt')
wavelength_AG, n_AG_real, n_AG_imag = data_AG.T

spline_AG_real = CubicSpline(wavelength_AG, n_AG_real)
spline_AG_imag = CubicSpline(wavelength_AG, n_AG_imag)


def n_AG(wavelength):
    wavelength /= 1000
    return spline_AG_real(wavelength) - spline_AG_imag(wavelength) * 1j

################################## Al2O3 DATA ###################################

data_Al2O3 = np.loadtxt('../data/Al2O3.txt')
wavelength_Al2O3, n_Al2O3_real, n_Al2O3_imag = data_Al2O3.T

spline_Al2O3_real = CubicSpline(wavelength_Al2O3, n_Al2O3_real)
spline_Al2O3_imag = CubicSpline(wavelength_Al2O3, n_Al2O3_imag)


def n_Al2O3(wavelength):
    wavelength /= 1000
    return spline_Al2O3_real(wavelength) - spline_Al2O3_imag(wavelength) * 1j

################################## Ti DATA ####################################

data_Ti = np.loadtxt('../data/Ag.txt')
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


##################################### main code #########################################

angle = np.linspace(65, 68, 600)
wavelength = np.linspace( 700,  800, 600)

result = np.zeros((len(wavelength),len(angle)))

thickness_layer = 20.0
thickness_au = 50
thickness_ti= 1.5

n_layer = 1.33
n_buffer = 1.33

for i in range(len(wavelength)):
    for j in range(len(angle)):
        n_gold = n_AU(wavelength[i])
        n_ti = n_Ti(wavelength[i])

        n = [1.51, n_ti, n_gold, n_layer, n_buffer]
        L = [thickness_ti * n_ti/ wavelength[i], thickness_au * n_gold / wavelength[i], thickness_layer * n_layer / wavelength[i]]

        gamma_return = gamma(n, L, angle[j],'tm')

        result[i][j] = np.abs(gamma_return) ** 2

plt.figure()
plt.pcolormesh(angle, wavelength ,result)
plt.xlabel("Angle [°]")
plt.ylabel("wavelength [nm]")
plt.title("50 nm Gold with "+str(thickness_ti)+" nm Ti")
plt.colorbar()
plt.show()

"""
angle = np.linspace(60, 70, 400)
wavelength = np.linspace( 600,  1600, 400)

result_1 = np.zeros((len(wavelength),len(angle)))
result_2 = np.zeros((len(wavelength),len(angle)))

thickness_layer = 20.0
thickness_au_1 = 0
thickness_ti_1= 46.93

thickness_au_2 = 50
thickness_ti_2= 1.5

n_layer = 1.33
n_buffer = 1.33

for i in range(len(wavelength)):
    for j in range(len(angle)):
        n_gold = n_AG(wavelength[i])
        n_ti = n_Ti(wavelength[i])

        n = [1.51, n_ti, n_gold, n_layer, n_buffer]
        L_1 = [thickness_ti_1 * n_ti/ wavelength[i], thickness_au_1 * n_gold / wavelength[i], thickness_layer * n_layer / wavelength[i]]
        L_2 = [thickness_ti_2 * n_ti/ wavelength[i], thickness_au_2 * n_gold / wavelength[i], thickness_layer * n_layer / wavelength[i]]

        gamma_return_1 = gamma(n, L_1, angle[j],'tm')
        gamma_return_2 = gamma(n, L_2, angle[j],'tm')

        result_1[i][j] = np.abs(gamma_return_1) ** 2
        result_2[i][j] = np.abs(gamma_return_2) ** 2


plt.figure()
plt.pcolormesh(angle, wavelength ,result_2-result_1, cmap='seismic')
plt.xlabel("Angle")
plt.ylabel("wavelength")
plt.title("Difference between Ti chip and 50 nm Au chip")
plt.colorbar()
plt.show()
"""