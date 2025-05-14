from matplotlib import pyplot as plt
from scipy.optimize import minimize, brute
import numpy as np

from solver.solver import intensity

from tqdm import tqdm  # Import the tqdm library


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


def spectral_sensitivity_surface(angle: float,
                                thickness: float,
                                wavelength: float,
                                surface_n,
                                wavelength_bounds: tuple = (1000, 2000),
                                surface_dn: float = 0.003,
                                bulk_n: float = 1.33,
                                surface_thickness: float = 10,
                                material: str = 'Au',
                                plot: bool = False,
                                optimize: bool = False,
                                center_sensitivity: bool = False):
    """
    Calculates spectral sensitivity for an array of surface refractive indices (surface_ns)
    and returns minimum wavelengths and sensitivities.

    Parameters:
    - angle (float): Incident angle.
    - thickness (float): Thickness of the layer.
    - wavelength (float): Wavelength to evaluate.
    - surface_ns (iterable): Array or list of surface refractive indices.
    - wavelength_bounds (tuple): The range of wavelengths to consider (min, max).
    - surface_dn (float): Change in refractive index for calculating sensitivity.
    - bulk_n (float): Bulk refractive index.
    - surface_thickness (float): Surface layer thickness.
    - material (str): The material being analyzed.

    Returns:
    - min_wavelengths_0 (np.ndarray): Array of minimum wavelengths for surface_n.
    - min_wavelengths_1 (np.ndarray): Array of minimum wavelengths for surface_n + surface_dn.
    - sensitivities (np.ndarray): Array of spectral sensitivities.
    """

    wavelength_range = np.arange(wavelength_bounds[0], wavelength_bounds[1], 0.005)   # Generate wavelength range

    sensitivities = []  # Initialize list to store sensitivities for each surface_n
    min_wavelengths_0 = []  # To store min wavelength values for surface_n
    min_wavelengths_1 = []  # To store min wavelength values for surface_n + surface_dn

    for surface_ni in tqdm(surface_n, desc="Calculating Sensitivity"):
        # Calculate intensities for surface_n and surface_n + surface_dn
        intensity_0 = intensity(angle, wavelength_range, thickness, bulk_n, n_layers=[surface_ni], thickness_layers=[surface_thickness], material=material)
        intensity_1 = intensity(angle, wavelength_range, thickness, bulk_n, n_layers=[surface_ni + surface_dn], thickness_layers=[surface_thickness], material=material)

        # Find minimum intensity wavelengths
        min_wavelength_0 = wavelength_range[np.argmin(intensity_0)]
        min_wavelength_1 = wavelength_range[np.argmin(intensity_1)]

        # Store the min wavelengths
        min_wavelengths_0.append(min_wavelength_0)
        min_wavelengths_1.append(min_wavelength_1)

        # Compute the spectral sensitivity
        spectral_sensitivity = (min_wavelength_1 - min_wavelength_0) / surface_dn
        sensitivities.append(spectral_sensitivity)  # Append sensitivity to list


    if plot:
        # Create main axis for intensity
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(surface_n, min_wavelengths_0, label=f'Original wavelength', color='blue')
        ax1.plot(surface_n, min_wavelengths_1, label=f'Response', color='red')
        ax1.set_xlabel('Refractive index [RIU]')
        ax1.set_ylabel('Wavelength responses [nm]')
        ax1.tick_params(axis='y')
        ax1.legend(loc='center left')
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(surface_n, sensitivities, label=f'Surface Spectral Sensitivity', color='black', linestyle='--')
        ax2.set_ylabel('Sensitivity [nm/RIU]', color='black')
        ax2.legend(loc='center right')

        # Add title including angle, thickness, and refractive index
        plt.title(f'Surface Spectral Sensitivity \nAngle: {angle:.2f}°, Thickness: {thickness:.2f} nm, Material: {material}')
        plt.show()

    return sensitivities, min_wavelengths_0, min_wavelengths_1

def spectral_sensitivity_layers(angle: float,
                                thickness: float,
                                wavelength_bounds: tuple = (800, 3000),
                                bulk_n: float = 1.33,
                                layer_n: float = 1.34,
                                layers_thickness: float = 25,
                                num_layers: int = 50,
                                material: str = 'Au',
                                plot: bool = False):
    """
    Calculate and plot spectral sensitivity as layers are incrementally added,
    based on the shift in the wavelength of minimum intensity.

    Args:
        angle (float): Incident angle in degrees.
        thickness (float): Base layer thickness (in nm).
        wavelength_bounds (tuple): Min and max bounds for wavelength (in nm).
        bulk_n (float): Refractive index of the bulk medium.
        surface_n (float): Refractive index of the added layers.
        surface_thickness (float): Thickness of each added layer (in nm).
        num_layers (int): Total number of layers to add incrementally.
        material (str): Material name (e.g., 'Au', 'Ag').
        plot (bool): Flag to enable plotting.

    Returns:
        Tuple[List[float], List[float]]: Sensitivity values and minimum wavelengths recorded
        across the added layers.
    """
    # Generate wavelength range
    wavelength_range = np.arange(wavelength_bounds[0], wavelength_bounds[1], 0.005)

    # Initialize lists to store cumulative layer data, sensitivities, and minimum wavelengths
    cumulative_n_layers = []
    cumulative_thickness_layers = []
    sensitivities = []
    min_wavelengths = []

    # Loop through adding layers incrementally
    for num in tqdm(range(1, num_layers + 1), desc="Adding Layers"):
        # Add a new layer with the specified refractive index and thickness
        cumulative_n_layers.append(layer_n)
        cumulative_thickness_layers.append(layers_thickness)

        # Calculate intensity without and with the current cumulative configuration
        intensity_without_layer = intensity(
            angle, wavelength_range, thickness, bulk_n,
            n_layers=cumulative_n_layers[:-1],  # Exclude current layer
            thickness_layers=cumulative_thickness_layers[:-1],  # Exclude current layer
            material=material
        )
        intensity_with_layer = intensity(
            angle, wavelength_range, thickness, bulk_n,
            n_layers=cumulative_n_layers,  # Include current layer
            thickness_layers=cumulative_thickness_layers,  # Include current layer
            material=material
        )

        # Find the wavelengths that correspond to the minimum intensity
        min_wavelength_without_layer = wavelength_range[np.argmin(intensity_without_layer)]
        min_wavelength_with_layer = wavelength_range[np.argmin(intensity_with_layer)]

        # Sensitivity is the change in minimum intensity wavelength after adding the layer
        sensitivity = np.abs(min_wavelength_with_layer - min_wavelength_without_layer)
        sensitivities.append(sensitivity)
        min_wavelengths.append(min_wavelength_with_layer)  # Record the minimum wavelength

    # Plot results
    if plot:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot sensitivities on the second axis
        ax1.set_xlabel('Number of Layers')
        ax1.set_ylabel('Wavelength of Min Intensity [nm]', color='blue')
        ax1.plot(range(1, num_layers + 1), min_wavelengths, marker='o', color='blue', label='Min Intensity Wavelength')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True)

        # Create a second y-axis for sensitivities
        ax2 = ax1.twinx()
        ax2.set_ylabel('Spectral Sensitivity [nm]', color='black')
        ax2.plot(range(1, num_layers + 1), sensitivities, marker='s', linestyle='--', color='black',
                 label='Spectral Sensitivity')
        ax2.tick_params(axis='y', labelcolor='black')

        # Add a combined legend
        fig.legend(loc='center right', bbox_to_anchor=(0.9, 0.5))
        plt.title(f'Spectral Sensitivity and Min Wavelength vs. Number of Layers\nAngle: {angle:.2f}°, Thickness: {thickness:.2f} nm, Material: {material}')
        plt.show()

    return sensitivities, min_wavelengths



if __name__ == "__main__":
    # Example usage
    wavelength = np.linspace(1500, 1700, 100)
    bulk_n = np.linspace(1.33, 1.343, 100)
    surface_n = np.linspace(1.33, 1.6, 100)

    #imaging_sensitivity_bulk(angle=62.5, wavelength=wavelength, thickness=50, material='Ag', plot=True, optimize=True)
    #imaging_sensitivity_surface(angle=62.5, wavelength=wavelength, thickness=50, material='Ag', plot=True, optimize=True)
    #spectral_sensitivity_bulk(angle=63.5, wavelength=1500, bulk_n=bulk_n, thickness=50, material='Ag', plot=True, optimize=False)
    #spectral_sensitivity_surface(angle=62.5, wavelength=1500, surface_n=surface_n, thickness=50, material='Ag', plot=True, optimize=False)
    spectral_sensitivity_layers(angle=63, thickness=50, material='Ag', plot=True)




