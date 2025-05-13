import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import seaborn as sns

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



############################################ wavelength sweep ##########################################################

angle0 = 66.1
wavelength0 = np.linspace(500, 900, 1000)
result0 = np.zeros(len(wavelength0))

thickness_au = 50.0
thickness_ti = 1.5

for j in range(len(wavelength0)):
    n_gold = n_AU(wavelength0[j])
    n_Titan = n_Ti(wavelength0[j])
    gamma_return = gamma([1.51, n_gold, n_Titan, 1.33],[thickness_au * n_gold / wavelength0[j], thickness_ti * n_Titan / wavelength0[j]], angle0,'tm')

    result0[j] = np.abs(gamma_return) ** 2

##############################
angle1 = 62.54
wavelength1 = np.linspace(1000, 2000, 1000)
result1 = np.zeros(len(wavelength1))

thickness_au1 = 50.0
thickness_ti = 1.5

for j in range(len(wavelength1)):
    n_gold = n_AU(wavelength1[j])
    n_Titan = n_Ti(wavelength1[j])
    gamma_return = gamma([1.51, n_gold, n_Titan, 1.33],[thickness_au1 * n_gold / wavelength1[j], thickness_ti * n_Titan / wavelength1[j]], angle1,'tm')

    result1[j] = np.abs(gamma_return) ** 2

##############################
angle2 = 62.58
wavelength2 = np.linspace(1000, 2000, 1000)
result2 = np.zeros(len(wavelength2))

thickness_au2 = 40.0
thickness_ti = 1.5

for j in range(len(wavelength2)):
    n_gold = n_AU(wavelength2[j])
    n_Titan = n_Ti(wavelength2[j])
    gamma_return = gamma([1.51, n_gold, n_Titan, 1.33],[thickness_au2 * n_gold / wavelength2[j], thickness_ti * n_Titan / wavelength2[j]], angle2,'tm')

    result2[j] = np.abs(gamma_return) ** 2

##############################
angle3 = 62.69
wavelength3 = np.linspace(1000, 2000, 1000)
result3 = np.zeros(len(wavelength3))

thickness_au3 = 30.0
thickness_ti = 1.5

for j in range(len(wavelength3)):
    n_gold = n_AU(wavelength3[j])
    n_Titan = n_Ti(wavelength3[j])
    gamma_return = gamma([1.51, n_gold, n_Titan, 1.33],[thickness_au3 * n_gold / wavelength3[j], thickness_ti * n_Titan / wavelength3[j]], angle3,'tm')

    result3[j] = np.abs(gamma_return) ** 2

#####################
plt.figure(figsize=(8, 6), dpi=120)

plt.tight_layout()

plt.plot(wavelength0, result0, label='750 nm,  50 nm Gold, '+str(angle0)+'°', c='#1f77b4')
plt.plot(wavelength1, result1, label='1540 nm, 50 nm Gold, '+str(angle1)+'°', c='#ff7f0e' )
plt.plot(wavelength2, result2, label='1540 nm, 40 nm Gold, '+str(angle2)+'°', c='#2ca02c')
plt.plot(wavelength3, result3, label='1540 nm, 30 nm Gold, '+str(angle3)+'°', c='#d62728')

plt.vlines(x=1540, ymin=0, ymax=1, colors='black', ls=':', lw=2)

plt.ylabel("intensity")
plt.xlabel("wavelength [nm]")
#plt.legend(loc='upper left')

plt.xticks(np.arange(250, 2100, 250))
plt.yticks(np.arange(0, 1.01, 0.2));

plt.xticks(np.arange(250, 2100, 125), minor=True)   # set minor ticks on x-axis
plt.yticks(np.arange(0, 1, 0.1), minor=True)   # set minor ticks on y-axis
plt.tick_params(which='minor', length=0)       # remove minor tick lines

plt.grid()                                     # draw grid for major ticks
plt.grid(which='minor', alpha=0.3)             # draw grid for minor ticks on x-axis

plt.show()

############################################ angle sweep ##########################################################

angle0 = np.linspace(62, 68, 1000)
wavelength0 = 750
result0 = np.zeros(len(angle0))

thickness_au = 50.0
thickness_ti = 1.5

for j in range(len(angle0)):
    n_gold = n_AU(wavelength0)
    n_Titan = n_Ti(wavelength0)
    gamma_return = gamma([1.51, n_gold, n_Titan, 1.33],[thickness_au * n_gold / wavelength0, thickness_ti * n_Titan / wavelength0], angle0[j],'tm')

    result0[j] = np.abs(gamma_return) ** 2

##############################
angle1 = np.linspace(62, 64, 1000)
wavelength1 = 1540
result1 = np.zeros(len(angle1))

thickness_au1 = 50.0
thickness_ti = 1.5

for j in range(len(angle1)):
    n_gold = n_AU(wavelength1)
    n_Titan = n_Ti(wavelength1)
    gamma_return = gamma([1.51, n_gold, n_Titan, 1.33],[thickness_au1 * n_gold / wavelength1, thickness_ti * n_Titan / wavelength1], angle1[j],'tm')

    result1[j] = np.abs(gamma_return) ** 2

##############################
angle2 = np.linspace(62, 64, 1000)
wavelength2 = 1540
result2 = np.zeros(len(angle2))

thickness_au2 = 40.0
thickness_ti = 1.5

for j in range(len(angle2)):
    n_gold = n_AU(wavelength2)
    n_Titan = n_Ti(wavelength2)
    gamma_return = gamma([1.51, n_gold, n_Titan, 1.33],[thickness_au2 * n_gold / wavelength2, thickness_ti * n_Titan / wavelength2], angle2[j],'tm')

    result2[j] = np.abs(gamma_return) ** 2

##############################
angle3 = np.linspace(62, 64, 1000)
wavelength3 = 1540
result3 = np.zeros(len(angle3))

thickness_au3 = 30.0
thickness_ti = 1.5

for j in range(len(angle3)):
    n_gold = n_AU(wavelength3)
    n_Titan = n_Ti(wavelength3)
    gamma_return = gamma([1.51, n_gold, n_Titan, 1.33],[thickness_au3 * n_gold / wavelength3, thickness_ti * n_Titan / wavelength3], angle3[j],'tm')

    result3[j] = np.abs(gamma_return) ** 2

#####################
plt.figure(figsize=(8, 6), dpi=120)

plt.tight_layout()

plt.plot(angle0, result0, label='750 nm,  50 nm Gold', c='#1f77b4')
plt.plot(angle1, result1, label='1540 nm, 50 nm Gold', c='#ff7f0e' )
plt.plot(angle2, result2, label='1540 nm, 40 nm Gold', c='#2ca02c')
plt.plot(angle3, result3, label='1540 nm, 30 nm Gold', c='#d62728')

plt.ylabel("intensity")
plt.xlabel("angle [°]")
#plt.legend(loc='upper left')

plt.xticks(np.arange(62, 68, 1))
plt.yticks(np.arange(0, 1.01, 0.2))

plt.xticks(np.arange(62, 68, 0.2), minor=True)   # set minor ticks on x-axis
plt.yticks(np.arange(0, 1, 0.2), minor=True)   # set minor ticks on y-axis
plt.tick_params(which='minor', length=0)       # remove minor tick lines

plt.grid()                                     # draw grid for major ticks
plt.grid(which='minor', alpha=0.3)             # draw grid for minor ticks on x-axis

plt.show()


############################################ RI sweep ##########################################################
n_water = np.linspace(1.32, 1.34, 1000)

angle0 = 66.1
wavelength0 = 750
result0 = np.zeros(len(n_water))

thickness_au = 50.0
thickness_ti = 1.5

for j in range(len(n_water)):
    n_gold = n_AU(wavelength0)
    n_Titan = n_Ti(wavelength0)
    gamma_return = gamma([1.51, n_gold, n_Titan, n_water[j]],[thickness_au * n_gold / wavelength0, thickness_ti * n_Titan / wavelength0], angle0,'tm')

    result0[j] = np.abs(gamma_return) ** 2

##############################
angle1 = 62.54
wavelength1 = 1540
result1 = np.zeros(len(n_water))

thickness_au1 = 50.0
thickness_ti = 1.5

for j in range(len(n_water)):
    n_gold = n_AU(wavelength1)
    n_Titan = n_Ti(wavelength1)
    gamma_return = gamma([1.51, n_gold, n_Titan, n_water[j]],[thickness_au1 * n_gold / wavelength1, thickness_ti * n_Titan / wavelength1], angle1,'tm')

    result1[j] = np.abs(gamma_return) ** 2

##############################
angle2 = 62.58
wavelength2 = 1540
result2 = np.zeros(len(n_water))

thickness_au2 = 40.0
thickness_ti = 1.5

for j in range(len(n_water)):
    n_gold = n_AU(wavelength2)
    n_Titan = n_Ti(wavelength2)
    gamma_return = gamma([1.51, n_gold, n_Titan, n_water[j]],[thickness_au2 * n_gold / wavelength2, thickness_ti * n_Titan / wavelength2], angle2,'tm')

    result2[j] = np.abs(gamma_return) ** 2

##############################
angle3 = 62.69
wavelength3 = 1540

result3 = np.zeros(len(n_water))

thickness_au3 = 30.0
thickness_ti = 1.5

for j in range(len(n_water)):
    n_gold = n_AU(wavelength3)
    n_Titan = n_Ti(wavelength3)
    gamma_return = gamma([1.51, n_gold, n_Titan, n_water[j]],[thickness_au3 * n_gold / wavelength3, thickness_ti * n_Titan / wavelength3], angle3,'tm')

    result3[j] = np.abs(gamma_return) ** 2

#####################
plt.figure(figsize=(8, 6), dpi=120)

plt.tight_layout()

plt.plot(n_water, result0, label='750 nm,  50 nm Gold', c='#1f77b4')
plt.plot(n_water, result1, label='1540 nm, 50 nm Gold', c='#ff7f0e' )
plt.plot(n_water, result2, label='1540 nm, 40 nm Gold', c='#2ca02c')
plt.plot(n_water, result3, label='1540 nm, 30 nm Gold', c='#d62728')

plt.ylabel("intensity")
plt.xlabel("RI [RIU]")
#plt.legend(loc='upper left')


plt.xticks(np.arange(1.32, 1.34, 0.005))
plt.yticks(np.arange(0, 1.01, 0.2))

plt.xticks(np.arange(1.32, 1.34, 0.002), minor=True)   # set minor ticks on x-axis
plt.yticks(np.arange(0, 1, 0.2), minor=True)   # set minor ticks on y-axis
plt.tick_params(which='minor', length=0)       # remove minor tick lines

plt.grid()                                     # draw grid for major ticks
plt.grid(which='minor', alpha=0.3)             # draw grid for minor ticks on x-axis


plt.show()
