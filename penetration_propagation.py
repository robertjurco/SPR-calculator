import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from materials.material import material_eps


def calculate(wavelengths, material):
    # Get the dielectric constant for all wavelengths at once (vectorized)
    eps_ = material_eps(material, wavelengths)

    # Static buffer value
    eps_buffer = np.sqrt(1.33)

    # Calculate gamma_d (complex propagation constant)
    gamma_d = 1j * 2 * np.pi / wavelengths * eps_buffer / np.sqrt(eps_ + eps_buffer)

    # Penetration depth (real part of gamma_d)
    pen_depth = 1 / np.abs(gamma_d.real)

    # Calculate beta (complex propagation parameter)
    beta = 2 * np.pi / wavelengths * np.sqrt(eps_ * eps_buffer / (eps_ + eps_buffer))

    # Propagation length (imaginary part of beta)
    prop_length = 1 / (2 * beta.imag)

    return prop_length, pen_depth


def plot_materials(wavelengths, materials, loglog = True, colormap = 'nipy_spectral'):
    # Prepare plots
    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)

    norm = plt.Normalize(vmin=np.min(wavelengths), vmax=np.max(wavelengths))
    cmap = matplotlib.colormaps.get_cmap(colormap)

    for material in materials:
        prop_length, pen_depth = calculate(wavelengths, material)

        for i in range(len(wavelengths) - 1):
            ax.plot(
                prop_length[i:i + 2]/1000,
                pen_depth[i:i + 2]/1000,
                color=cmap(norm(wavelengths[i])),
                lw=2,
            )

        # Add the material name as a label at the end of the line
        ax.text(prop_length[0]/1000-0.02, pen_depth[0]/1000, material,
                va='center', ha='right', alpha=1)

    # Finalize plots
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label="Wavelength [nm]")

    ax.set_title("Normalized Penetration Depth & Propagation Length with Gradient", fontsize=14)
    ax.set_xlabel("Propagation Length [$\mu$m]", fontsize=12)
    ax.set_ylabel("Penetration Depth [$\mu$m]", fontsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    if loglog:
        ax.set_xscale('log')
        ax.set_yscale('log')

    plt.tight_layout()
    plt.show()


################################### MAIN CODE ###########################################

# Define wavelengths
wavelengths = np.linspace(500, 1700, 1000)  # Wavelengths in nm

# Select materials
materials = ['Au', 'Cu', 'Ag', 'Al', 'Rh', 'TiN', 'ZrN', 'test']

# Plot
plot_materials(wavelengths, materials)
