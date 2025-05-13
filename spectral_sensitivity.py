import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from materials.material import material_n


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

def intensity(angle, wavelength, thickness, RI, material='Au'):
    thickness_ti = 1.5
    n_buffer = RI
    n_mat = np.conjugate(material_n(material, wavelength))
    n_ti = np.conjugate(material_n('Ti', wavelength))
    n = [1.55, n_ti, n_mat, n_buffer]
    L = [thickness_ti * n_ti / wavelength, thickness * n_mat / wavelength]
    gamma_return = gamma(n, L, angle, 'tm')
    result = np.abs(gamma_return) ** 2
    return result


# Function to calculate intensity profile for given angle and thickness
def intensity_profile(angle, thickness, RI, material='Au'):
    wavelengths = np.linspace(500,2000, 200)  # Range: wavelength-100 nm to wavelength+100 nm
    intensities = np.array([intensity(angle, w, thickness, RI, material) for w in wavelengths])
    #plt.figure()
    #plt.plot(wavelengths, intensities)
    #plt.show()
    return wavelengths, intensities


# Function to calculate spectral sensitivity for given angle and thickness
def spectral_sensitivity(angle, thickness, RI, delta_RI, material='Au'):
    wavelengths_RI, intensities_RI = intensity_profile(angle, thickness, RI, material)
    wavelengths_RI_delta, intensities_RI_delta = intensity_profile(angle, thickness, RI + delta_RI,
                                                                   material)

    # Find the minima of both intensity profiles
    min_idx_RI = np.argmin(intensities_RI)
    min_idx_RI_delta = np.argmin(intensities_RI_delta)

    # Calculate the wavelength shift of the minima (this is the sensitivity)
    wavelength_shift = wavelengths_RI_delta[min_idx_RI_delta] - wavelengths_RI[min_idx_RI]

    # Spectral sensitivity is the shift divided by the change in RI
    sensitivity = wavelength_shift / delta_RI
    return sensitivity, wavelengths_RI[min_idx_RI]


# Function to plot spectral sensitivity as a function of RI for different angles
def plot_sensitivity_for_RI(thickness, delta_RI, RI_start, RI_end, num_points, material='Au',
                            fixed_parameter='thickness'):
    # Set up an array of RI values
    RI_values = np.linspace(RI_start, RI_end, num_points)

    # Array to hold sensitivities for each angle
    sensitivities = []
    dip_positions = []

    # Angle values to iterate over (fixed parameter)
    angles = [62.5, 65, 67, 69]  # Example angles to plot

    # Iterate over the angles (or thickness if fixed)
    for angle in angles:
        angle_sensitivities = []
        angle_dip_positions = []

        # Calculate spectral sensitivity for each RI value at the current angle
        for RI in RI_values:
            print(RI)
            sensitivity, dip_position = np.fabs(spectral_sensitivity(angle, thickness, RI, delta_RI, material))
            angle_sensitivities.append(sensitivity)
            angle_dip_positions.append(dip_position)

        # Add the calculated sensitivities to the list
        sensitivities.append(angle_sensitivities)
        dip_positions.append(angle_dip_positions)

    # Plot the spectral sensitivity as a function of RI for each angle
    plt.figure(figsize=(10, 6))
    for i, angle_sensitivities in enumerate(sensitivities):
        plt.plot(dip_positions[i], angle_sensitivities, label=f'Angle = {angles[i]}°')

    plt.xlabel('Dip Position (Wavelength, nm)')
    plt.ylabel('Spectral Sensitivity (nm/RI)')
    plt.title('Spectral Sensitivity vs. Refractive Index for Different Angles')
    plt.legend()
    plt.grid(True)
    plt.show()



########################################################################################################################


# Example usage:
RI_start = 1.33  # Starting refractive index
RI_end = 1.4  # Ending refractive index
num_points = 10  # Number of points to calculate between RI_start and RI_end
delta_RI = 0.01  # Change in RI for sensitivity calculation
thickness = 42  # Fixed thickness in nm

# Plot the sensitivity
plot_sensitivity_for_RI(thickness, delta_RI, RI_start, RI_end, num_points)