import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from materials.material import material_eps

def plot_materials(wavelengths, materials):
    # Prepare plots: 1 row, 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=120)

    for material in materials:
        eps_real = np.zeros(len(wavelengths))
        eps_imag = np.zeros(len(wavelengths))

        for i in range(len(wavelengths)):
            eps = material_eps(material, wavelengths[i])
            eps_real[i] = np.real(eps)
            eps_imag[i] = np.imag(eps)

        ax1.plot(wavelengths, eps_real, label=material)
        ax2.plot(wavelengths, eps_imag, label=material)

    # Plot for real part
    ax1.set_title("Real part of $\epsilon$", fontsize=14)
    ax1.set_xlabel("Wavelength", fontsize=12)
    ax1.set_ylabel("Real $\epsilon$", fontsize=12)
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax1.legend()

    # Plot for imaginary part
    ax2.set_title("Imaginary part of $\epsilon$", fontsize=14)
    ax2.set_xlabel("Wavelength", fontsize=12)
    ax2.set_ylabel("Imag $\epsilon$", fontsize=12)
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.show()

################################### MAIN CODE ###########################################

# Define wavelengths
wavelengths = np.linspace(500, 1700, 1000)  # Wavelengths in nm

# Select materials
materials = ['Au', 'Cu', 'Ag', 'Al', 'Rh', 'TiN', 'ZrN', 'test']

# Plot
plot_materials(wavelengths, materials)
