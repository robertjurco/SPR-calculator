from matplotlib import pyplot as plt
from scipy.optimize import minimize, brute
import numpy as np

from solver.solver import intensity

# Function to simplify float formatting dynamically
def format_float(value: float) -> str:
    """Dynamically format a float to remove trailing zeros."""
    return f"{value:.15f}".rstrip('0').rstrip('.')

##################################### Imaging Sensitivity #########################################

def imaging_sensitivity_bulk(angle: float,
                             wavelength,
                             thickness: float,
                             bulk_n: float = 1.33,
                             bulk_dn: float = 0.00005,
                             material: str = 'Au',
                             plot: bool = False,
                             optimize: bool = False,
                             center_sensitivity: bool = True):
    """
    Calculate imaging bulk sensitivity for a change in index of refraction by bulk_dn with optional plotting and optimization.
    Args:
        angle (float): Incident angle in degrees.
        wavelength: Wavelength in nm (array or single value).
        thickness (float): Thickness in nm.
        bulk_n (float): Refractive index of the bulk medium.
        bulk_dn (float): Differential refractive index change for bulk sensitivity.
        material (str): Material name (e.g., 'Au', 'Ag').
        plot (bool): Flag to enable plotting.
        optimize (bool): Flag to enable optimization of angle and thickness.
        center_sensitivity (bool): If True, centers around the maximum and then adjusts to midpoint between max and min sensitivity.


    Returns:
        float: Sensitivity for the bulk material.
    """
    # Validate and pre-process wavelength
    if isinstance(wavelength, (list, np.ndarray)):
        # If wavelength is an array, convert it to a scalar for optimization
        wavelength_range = (wavelength[0], wavelength[-1])  # Save bounds
        wavelength_mean = np.mean(wavelength)  # Use the mean as the initial guess for optimization
    else:
        # Scalar wavelength case
        wavelength_range = (400, 800)  # Default range, adjustable as needed (or you can skip this entirely)
        wavelength_mean = wavelength

    def objective(x):
        """Objective function for optimization."""
        ang, wav, thick = x
        # Compute sensitivity without recursive optimization
        intensity_0 = intensity(ang, wav, thick, bulk_n, material=material)
        intensity_1 = intensity(ang, wav, thick, bulk_n + bulk_dn, material=material)
        # Sensitivity
        sensitivity = (intensity_1 - intensity_0) / bulk_dn
        return -sensitivity  # Negate to optimize for maximum sensitivity

    if optimize:
        # Step 1: Perform brute-force optimization for a rough global search
        ranges = [(60, 70), wavelength_range, (30, 100)]  # Define the ranges for angle, wavelength, and thickness
        brute_result = brute(objective, ranges=ranges, Ns=20, full_output=True, finish=None)  # Ns is the grid resolution

        # Extract the parameters found by brute
        starting_point = brute_result[0]

        # Step 2: Refine the result with minimize using the brute result as starting point
        res = minimize(objective, x0=starting_point, bounds=ranges, method='Nelder-Mead')
        angle, wavelength_mean, thickness = res.x

    intensity_0 = intensity(angle, wavelength, thickness, bulk_n, material=material)
    intensity_1 = intensity(angle, wavelength, thickness, bulk_n+bulk_dn, material=material)

    # Calculate sensitivity
    sensitivity = (intensity_1 - intensity_0) / bulk_dn

    # Extend and center the data around the maximum sensitivity
    if center_sensitivity and isinstance(wavelength, np.ndarray):
        # Find index of the maximum sensitivity
        max_idx = np.argmax(sensitivity)
        max_wavelength = wavelength[max_idx]

        # Extend wavelength symmetrically around the max point
        step = wavelength[1] - wavelength[0]  # Assuming wavelength is linearly spaced
        half_range = len(wavelength) // 2

        extended_wavelength = np.arange(
            max_wavelength - step * half_range,
            max_wavelength + step * (half_range + 1),
            step
        )

        # Recalculate intensities and sensitivities for the new range
        intensity_0 = intensity(angle, extended_wavelength, thickness, bulk_n, material=material)
        intensity_1 = intensity(angle, extended_wavelength, thickness, bulk_n + bulk_dn, material=material)
        sensitivity = (intensity_1 - intensity_0) / bulk_dn

        # Prepare wavelength, intensity, and sensitivity for the next step
        wavelength = extended_wavelength

        # Step 2: Center around midpoint between max and min sensitivity
        min_idx = np.argmin(sensitivity)
        max_idx = np.argmax(sensitivity)

        # Find the midpoint wavelength between max and min
        min_wavelength = wavelength[min_idx]
        max_wavelength = wavelength[max_idx]
        midpoint_wavelength = (min_wavelength + max_wavelength) / 2

        # Extend wavelength symmetrically around the midpoint
        extended_wavelength = np.arange(
            midpoint_wavelength - step * half_range,
            midpoint_wavelength + step * (half_range + 1),
            step
        )

        # Recalculate intensities and sensitivities for the new midpoint-centered range
        intensity_0 = intensity(angle, extended_wavelength, thickness, bulk_n, material=material)
        intensity_1 = intensity(angle, extended_wavelength, thickness, bulk_n + bulk_dn, material=material)
        sensitivity = (intensity_1 - intensity_0) / bulk_dn

        # Update wavelength with the final extended range
        wavelength = extended_wavelength

    if plot:
        # Create main axis for intensity
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(wavelength, intensity_0, label=f'Intensity, Bulk RI={format_float(bulk_n)}', color='blue')
        ax1.plot(wavelength, intensity_1, label=f'Intensity, Bulk RI={format_float(bulk_n+bulk_dn)}', color='red')
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Intensity')
        ax1.tick_params(axis='y')
        ax1.legend(loc='lower left')
        ax1.grid(True)

        # Create secondary axis for sensitivity
        ax2 = ax1.twinx()
        ax2.plot(wavelength, sensitivity, label='Sensitivity', color='black', linestyle='--')
        ax2.set_ylabel('Sensitivity', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        ax2.legend(loc='lower right')

        # Add title including angle, thickness, and refractive index
        plt.title(f'Bulk Sensitivity and Intensities\nAngle: {angle:.2f}°, Thickness: {thickness:.2f} nm, Material: {material}')
        plt.show()

    return sensitivity


def imaging_sensitivity_surface(angle: float,
                        wavelength,
                        thickness: float,
                        bulk_n: float = 1.33,
                        surface_thickness: float = 10,
                        surface_n: float = 1.33,
                        surface_dn: float = 0.01,
                        material: str = 'Au',
                        plot: bool = False,
                        optimize: bool = False,
                        center_sensitivity: bool = True):
    """
    Calculate surface sensitivity using changes in refractive index with optional plotting and optimization.

    Args:
        angle (float): Incident angle in degrees.
        wavelength: Wavelength in nm (array or single value).
        thickness (float): Thickness in nm.
        bulk_n (float): Refractive index of the bulk medium.
        surface_thickness (float): Thickness of the surface layer.
        surface_n (float): Refractive index of the surface layer.
        surface_dn (float): Differential refractive index change for surface sensitivity.
        material (str): Material name (e.g., 'Au', 'Ag').
        plot (bool): Flag to enable plotting.
        optimize (bool): Flag to enable optimization of angle and thickness.
        center_sensitivity (bool): If True, centers around the midpoint between max and min sensitivity.

    Returns:
        float: Calculated surface sensitivity.
    """

    # Validate and pre-process wavelength
    if isinstance(wavelength, (list, np.ndarray)):
        wavelength_range = (wavelength[0], wavelength[-1])
        wavelength_mean = np.mean(wavelength)
    else:
        wavelength_range = (400, 800)
        wavelength_mean = wavelength


    def objective(x):
        ang, wav, thick = x
        intensity_0 = intensity(ang, wav, thick, bulk_n, n_layers=[surface_n], thickness_layers=[surface_thickness], material=material)
        intensity_1 = intensity(ang, wav, thick, bulk_n, n_layers=[surface_n + surface_dn], thickness_layers=[surface_thickness], material=material)
        sensitivity = (intensity_1 - intensity_0) / surface_dn
        return -sensitivity  # Negate to maximize sensitivity

    if optimize:
        # Step 1: Perform brute-force optimization for a rough global search
        ranges = [(60, 70), wavelength_range, (30, 100)]  # Define the ranges for angle, wavelength, and thickness
        brute_result = brute(objective, ranges=ranges, Ns=20, full_output=True, finish=None)  # Ns is the grid resolution

        # Extract the parameters found by brute
        starting_point = brute_result[0]

        # Step 2: Refine the result with minimize using the brute result as starting point
        res = minimize(objective, x0=starting_point, bounds=ranges, method='Nelder-Mead')
        angle, wavelength_mean, thickness = res.x

    # Calculate intensities and sensitivity
    intensity_0 = intensity(angle, wavelength, thickness, bulk_n, n_layers=[surface_n], thickness_layers=[surface_thickness], material=material)
    intensity_1 = intensity(angle, wavelength, thickness, bulk_n, n_layers=[surface_n + surface_dn], thickness_layers=[surface_thickness], material=material)
    sensitivity = (intensity_1 - intensity_0) / surface_dn

    if center_sensitivity and isinstance(wavelength, np.ndarray):
        # Find index of the maximum sensitivity
        max_idx = np.argmax(sensitivity)
        max_wavelength = wavelength[max_idx]

        # Extend wavelength symmetrically around the max point
        step = wavelength[1] - wavelength[0]  # Assuming wavelength is linearly spaced
        half_range = len(wavelength) // 2

        extended_wavelength = np.arange(
            max_wavelength - step * half_range,
            max_wavelength + step * (half_range + 1),
            step
        )

        # Recalculate intensities and sensitivities for the new range
        intensity_0 = intensity(angle, wavelength, thickness, bulk_n, n_layers=[surface_n],
                                thickness_layers=[surface_thickness], material=material)
        intensity_1 = intensity(angle, wavelength, thickness, bulk_n, n_layers=[surface_n + surface_dn],
                                thickness_layers=[surface_thickness], material=material)
        sensitivity = (intensity_1 - intensity_0) / surface_dn

        # Prepare wavelength, intensity, and sensitivity for the next step
        wavelength = extended_wavelength

        # Step 2: Center around midpoint between max and min sensitivity
        min_idx = np.argmin(sensitivity)
        max_idx = np.argmax(sensitivity)

        # Find the midpoint wavelength between max and min
        min_wavelength = wavelength[min_idx]
        max_wavelength = wavelength[max_idx]
        midpoint_wavelength = (min_wavelength + max_wavelength) / 2

        # Extend wavelength symmetrically around the midpoint
        extended_wavelength = np.arange(
            midpoint_wavelength - step * half_range,
            midpoint_wavelength + step * (half_range + 1),
            step
        )

        # Recalculate intensities and sensitivities for the new midpoint-centered range
        intensity_0 = intensity(angle, wavelength, thickness, bulk_n, n_layers=[surface_n],
                                thickness_layers=[surface_thickness], material=material)
        intensity_1 = intensity(angle, wavelength, thickness, bulk_n, n_layers=[surface_n + surface_dn],
                                thickness_layers=[surface_thickness], material=material)
        sensitivity = (intensity_1 - intensity_0) / surface_dn

        # Update wavelength with the final extended range
        wavelength = extended_wavelength

    if plot:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(wavelength, intensity_0, label=f'Intensity, Surface RI={format_float(surface_n)}', color='blue')
        ax1.plot(wavelength, intensity_1, label=f'Intensity, Surface RI={format_float(surface_n + surface_dn)}',
                 color='red')
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Intensity')
        ax1.legend(loc='lower left')
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(wavelength, sensitivity, label='Sensitivity', color='black', linestyle='--')
        ax2.set_ylabel('Sensitivity', color='black')
        ax2.legend(loc='lower right')

        plt.title(f'Surface Sensitivity\nAngle: {angle:.2f}°, Thickness: {thickness:.2f} nm, Material: {material}')
        plt.show()

    return sensitivity


##################################### Spectral Sensitivity #########################################

def spectral_sensitivity_bulk(angle: float,
                             thickness: float,
                             wavelength: float,
                             bulk_n,
                             wavelength_bounds: tuple = (1000, 2000),
                             bulk_dn: float = 0.0005,
                             material: str = 'Au',
                             plot: bool = False,
                             optimize: bool = False,
                             center_sensitivity: bool = False):
    """
    Calculate spectral sensitivity for bulk material with optional plotting and optimization.
    """

    # Prepare wavelengths
    wavelength_range = np.arange(wavelength_bounds[0], wavelength_bounds[1], 0.1)

    intensity_0 = intensity(angle, wavelength_range, thickness, bulk_n, material=material)
    intensity_1 = intensity(angle, wavelength_range, thickness, bulk_n+bulk_dn, material=material)

    # Find minimum intensity along the wavelength axis for each bulk_n
    # find wavelengths corresponding to the minimum intensity
    min_wavelength_0 = wavelength_range[np.argmin(intensity_0, axis=1)]
    min_wavelength_1 = wavelength_range[np.argmin(intensity_1, axis=1)]


    # Calculate sensitivity
    spectral_sensitivity = (min_wavelength_1 - min_wavelength_0) / bulk_dn

    if plot:
        # Create main axis for intensity
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(bulk_n, min_wavelength_0, label=f'Original wavelength', color='blue')
        ax1.plot(bulk_n, min_wavelength_1, label=f'Response', color='red')
        ax1.set_xlabel('Refractive index [RIU]')
        ax1.set_ylabel('Wavelength responses [nm]')
        ax1.tick_params(axis='y')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(bulk_n, spectral_sensitivity, label=f'Bulk Spectral Sensitivity', color='black', linestyle='--')
        ax2.set_ylabel('Sensitivity [nm/RIU]', color='black')
        ax2.legend(loc='lower right')

        # Add title including angle, thickness, and refractive index
        plt.title(f'Bulk Spectral Sensitivity \nAngle: {angle:.2f}°, Thickness: {thickness:.2f} nm, Material: {material}')
        plt.show()

    return spectral_sensitivity


def spectral_sensitivity_surface(x, dn=0.00005, wavelength_range=(400, 800), resolution=1, plot: bool = False,
                                 optimize: bool = False):
    """
    Calculate spectral sensitivity for surface material with optional plotting and optimization.
    """
    if optimize:
        def objective(params):
            return -spectral_sensitivity_surface(params, dn=dn, wavelength_range=wavelength_range,
                                                 resolution=resolution)

        res = minimize(objective, x0=x, bounds=[(400, 800), (0, 90), (1, 100), (1, 100)])
        x = res.x

    center_wavelength, angle, thickness_au, thickness_ti = x
    wavelengths = np.arange(wavelength_range[0], wavelength_range[1], resolution)

    intensities = []
    for wavelength in wavelengths:
        intensities.append(plasmon_n((wavelength, angle, thickness_au, thickness_ti)))

    original_min_wavelength = wavelengths[np.argmin(intensities)]

    # Perturb the surface refractive index (apply dn)
    intensities_dn = []
    for wavelength in wavelengths:
        n_layer = 1.33 + dn
        n = [1.51, n_Ti(wavelength), n_AU(wavelength), n_layer, 1.33]
        L = [thickness_ti * n_Ti(wavelength) / wavelength, thickness_au * n_AU(wavelength) / wavelength, 0.0]
        intensity_dn = np.abs(gamma(n, L, angle, 'tm') ** 2)
        intensities_dn.append(intensity_dn)

    min_wavelength_dn = wavelengths[np.argmin(intensities_dn)]
    spectral_sensitivity = (min_wavelength_dn - original_min_wavelength) / dn

    if plot:
        plt.plot(wavelengths, intensities, label="Original")
        plt.plot(wavelengths, intensities_dn, label="With dn")
        plt.title('Spectral Sensitivity (Surface)')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # Example usage
    wavelength = np.linspace(1500, 1700, 100)
    bulk_n = np.linspace(1.33, 1.343, 100)

    #imaging_sensitivity_bulk(angle=62.5, wavelength=wavelength, thickness=50, material='Ag', plot=True, optimize=True)
    #imaging_sensitivity_surface(angle=62.5, wavelength=wavelength, thickness=50, material='Ag', plot=True, optimize=True)
    spectral_sensitivity_bulk(angle=63.5, wavelength=1500, bulk_n=bulk_n, thickness=50, material='Ag', plot=True, optimize=False)


