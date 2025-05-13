import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline


def n_(wavelength, filename):
    data = np.loadtxt(filename)
    wavelength_data, n_real, n_imag = data.T
    spline_real = CubicSpline(wavelength_data, n_real)
    spline_imag = CubicSpline(wavelength_data, n_imag)

    wavelength /= 1000
    return spline_real(wavelength) - spline_imag(wavelength) * 1j


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

def intensity(wavelength, angle, thickness, filename):
    result = np.zeros((len(wavelength), len(angle)))
    thickness_layer = 0.0
    thickness_ti = 1.5
    n_layer = 1.33
    n_buffer = 1.33
    for i in range(len(wavelength)):
        for j in range(len(angle)):
            n_mat = n_(wavelength[i], filename)
            n_ti = n_(wavelength[i], '../data/Ti.txt')
            n = [1.51, n_ti, n_mat, n_layer, n_buffer]
            L = [thickness_ti * n_ti / wavelength[i], thickness * n_mat / wavelength[i], thickness_layer * n_layer / wavelength[i]]
            gamma_return = gamma(n, L, angle[j], 'tm')
            result[i][j] = np.abs(gamma_return) ** 2
    return result


angle = np.linspace(60, 70, 50)
wavelength = np.linspace(700, 1700, 50)

result = intensity(wavelength, angle, 45, '../data/gold_yakubovsky.txt')

plt.figure()
plt.pcolormesh(angle, wavelength, result)
plt.xlabel("Angle [°]")
plt.ylabel("wavelength [nm]")
plt.title("Gold Intensity")
plt.colorbar()
plt.show()