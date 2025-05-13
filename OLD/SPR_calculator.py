import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

################################### CONSTANTS ####################################
c = 299792458
hbar = 6.5822e-16

################################## GOLD DATA ####################################

#data_AU = np.loadtxt('../data/gold_yakubovsky.txt')
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
"""
angle = 62.55
wavelength = 1540

n_fluid = np.linspace(1.325, 1.335, 500)
result = np.zeros(len(n_fluid))

thickness_au = 43
thickness_ti= 1.5



for i in range(len(n_fluid)):
    n_gold = n_AG(wavelength)
    n_ti = n_Ti(wavelength)

    n = [1.51, n_ti, n_gold, n_fluid[i]]
    L = [thickness_ti * n_ti/ wavelength, thickness_au * n_gold / wavelength]

    gamma_return = gamma(n, L, angle,'tm')

    result[i] = np.abs(gamma_return) ** 2

plt.figure()
plt.plot(n_fluid, result)
plt.xlabel("IR")
plt.ylabel("intensity")
plt.title("1540 nm with "+str(thickness_au)+" nm Gold")
plt.show()

"""

"""

angle = np.linspace(60, 70, 400)
wavelength = np.linspace( 600,  1600, 400)

result = np.zeros((len(wavelength),len(angle)))

thickness_layer = 20.0
thickness_au = 0
thickness_ti= 46.93

n_layer = 1.33
n_buffer = 1.33

for i in range(len(wavelength)):
    for j in range(len(angle)):
        n_gold = n_AG(wavelength[i])
        n_ti = n_Ti(wavelength[i])

        n = [1.51, n_ti, n_gold, n_layer, n_buffer]
        L = [thickness_ti * n_ti/ wavelength[i], thickness_au * n_gold / wavelength[i], thickness_layer * n_layer / wavelength[i]]

        gamma_return = gamma(n, L, angle[j],'tm')

        result[i][j] = np.abs(gamma_return) ** 2

plt.figure()
plt.pcolormesh(angle, wavelength ,result)
plt.xlabel("Angle")
plt.ylabel("wavelength")
plt.title("1540 nm with "+str(thickness_ti)+" nm Ti")
plt.colorbar()
plt.show()
"""
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

angle = np.linspace(62.3, 63, 400)
wavelength = 1540
result = np.zeros(len(angle))

thickness_au = 40.0
thickness_layer = 20.0

n_layer = 1.33
n_buffer = 1.33

for j in range(len(angle)):
    n_gold = n_AU(wavelength)
    gamma_return = gamma([1.51, n_gold, n_layer, n_buffer],[thickness_au * n_gold / wavelength, thickness_layer * n_layer / wavelength], angle[j],'tm')

    result[j] = np.abs(gamma_return) ** 2


angle2 = np.linspace(65, 68, 400)
wavelength2 = 750
result2 = np.zeros(len(angle))

thickness_au2 = 50.0
thickness_layer2 = 20.0

n_layer2 = 1.33
n_buffer2 = 1.33

for j in range(len(angle2)):
    n_gold = n_AU(wavelength2)
    gamma_return = gamma([1.51, n_gold, n_layer2, n_buffer2],[thickness_au2 * n_gold / wavelength2, thickness_layer2 * n_layer2 / wavelength2], angle2[j],'tm')

    result2[j] = np.abs(gamma_return) ** 2

angle3 = np.linspace(62.3, 63.4, 400)
wavelength3 = 1540
result3 = np.zeros(len(angle3))

thickness_au3 = 0.0
thickness_layer3 = 20.0

n_layer3 = 1.33
n_buffer3 = 1.33

for j in range(len(angle3)):
    n_gold = n_AU(wavelength3)
    gamma_return = gamma([1.51, n_gold, n_layer3, n_buffer3],[thickness_au3 * n_gold / wavelength3, thickness_layer3 * n_layer3 / wavelength3], angle3[j],'tm')

    result3[j] = np.abs(gamma_return) ** 2

#pcolormesh of interpolated uniform grid with log colormap

plt.figure()
plt.plot(angle, result, label='1540 nm - 40 nm Gold')
plt.plot(angle2, result2, label='750 nm - 50 nm Gold')
plt.plot(angle3, result3, label='1540 nm - 30 nm Gold')
plt.ylabel("intensity")
plt.xlabel("angle")
plt.legend()
plt.show()

"""
angle = 62.63
wavelength = np.linspace(1000, 2000, 1000)
result = np.zeros(len(wavelength))

thickness_au = 40.0
thickness_layer = 20.0

n_layer = 1.33
n_buffer = 1.33

for j in range(len(wavelength)):
    n_gold = n_AU(wavelength[j])
    gamma_return = gamma([1.51, n_gold, n_layer, n_buffer],[thickness_au * n_gold / wavelength[j], thickness_layer * n_layer / wavelength[j]], angle,'tm')

    result[j] = np.abs(gamma_return) ** 2


angle2 = 66.8
wavelength2 = np.linspace(300, 1000, 1000)
result2 = np.zeros(len(wavelength2))

thickness_au2 = 50.0
thickness_layer2 = 20.0

n_layer2 = 1.33
n_buffer2 = 1.33

for j in range(len(wavelength2)):
    n_gold = n_AU(wavelength2[j])
    gamma_return = gamma([1.51, n_gold, n_layer2, n_buffer2],[thickness_au2 * n_gold / wavelength2[j], thickness_layer2 * n_layer2 / wavelength2[j]], angle2,'tm')

    result2[j] = np.abs(gamma_return) ** 2

angle3 = 62.58
wavelength3 = np.linspace(1000, 2000, 1000)
result3 = np.zeros(len(wavelength3))

thickness_au3 = 50.0
thickness_layer3 = 20.0

n_layer3 = 1.33
n_buffer3 = 1.33

for j in range(len(wavelength3)):
    n_gold = n_AU(wavelength3[j])
    gamma_return = gamma([1.51, n_gold, n_layer3, n_buffer3],[thickness_au3 * n_gold / wavelength3[j], thickness_layer3 * n_layer3 / wavelength3[j]], angle3,'tm')

    result3[j] = np.abs(gamma_return) ** 2


angle4 = 62.78
wavelength4 = np.linspace(1000, 2000, 1000)
result4 = np.zeros(len(wavelength3))

thickness_au4 = 30.0
thickness_layer4 = 20.0

n_layer4 = 1.33
n_buffer4 = 1.33

for j in range(len(wavelength4)):
    n_gold = n_AU(wavelength4[j])
    gamma_return = gamma([1.51, n_gold, n_layer4, n_buffer4],[thickness_au4 * n_gold / wavelength4[j], thickness_layer4 * n_layer4 / wavelength4[j]], angle4,'tm')

    result4[j] = np.abs(gamma_return) ** 2

#pcolormesh of interpolated uniform grid with log colormap

plt.figure()
plt.plot(wavelength2, result2, label='750 nm, 50 nm Gold, '+str(angle2)+'°')
plt.plot(wavelength, result, label='1540 nm, 40 nm Gold, '+str(angle)+'°')
plt.plot(wavelength3, result3, label='1540 nm, 50 nm Gold, '+str(angle3)+'°')
plt.plot(wavelength4, result4, label='1540 nm, 30 nm Gold, '+str(angle4)+'°')

plt.vlines(x=1540, ymin=0, ymax=1, colors='black', ls=':', lw=2)

plt.ylabel("intensity")
plt.xlabel("wavelength")
plt.legend(loc='upper left')

plt.xticks(np.arange(250, 2100, 250))
plt.yticks(np.arange(0, 1.01, 0.2));

plt.xticks(np.arange(250, 2100, 125), minor=True)   # set minor ticks on x-axis
plt.yticks(np.arange(0, 1, 0.1), minor=True)   # set minor ticks on y-axis
plt.tick_params(which='minor', length=0)       # remove minor tick lines

plt.grid()                                     # draw grid for major ticks
plt.grid(which='minor', alpha=0.3)             # draw grid for minor ticks on x-axis

plt.show()
"""
"""
# Plot the 3D surface
fig = plt.figure()
angle, wavelength = np.meshgrid(angle, wavelength)
ax = plt.axes(projection ='3d')
ax.plot_wireframe(angle, wavelength, result, edgecolor='royalblue', lw=1)
plt.show()
"""
##################################### TESTS ###########################################

#plt.plot(wavelength_AU, n_AU_real)
#plt.plot(wavelength_AU, n_AU_imag)
#plt.show()


#################################### INTERFACE ######################################
"""
from matplotlib.widgets import Slider, Button, RadioButtons

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
t = np.arange(0.0, 1.0, 0.001)
a0 = 5
f0 = 3
s = a0*np.sin(2*np.pi*f0*t)
l, = plt.plot(t, s, lw=2, color='red')
plt.axis([0, 1, -10, 10])

axcolor = 'lightgoldenrodyellow'
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0)
samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)


def update(val):
    amp = samp.val
    freq = sfreq.val
    l.set_ydata(amp*np.sin(2*np.pi*freq*t))
    fig.canvas.draw_idle()
sfreq.on_changed(update)
samp.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    sfreq.reset()
    samp.reset()
button.on_clicked(reset)

rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)


def colorfunc(label):
    l.set_color(label)
    fig.canvas.draw_idle()
radio.on_clicked(colorfunc)

plt.show()
"""