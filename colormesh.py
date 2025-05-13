import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from solver.solver import intensity


def colormesh(material: str,
              fixed_params: dict,
              slice_ranges: dict,
              n_glass: float = 1.51,
              resolution: int = 100,
              plot: bool = True,
              n_layers: list = None,
              thickness_layers: list = None):
    """
    Slice the SPR (Surface Plasmon Resonance) spectrum by fixing two parameters and exploring the other two.
    Optionally plots the resulting spectrum.

    Args:
        material (str): Name of the material for SPR simulation.
        fixed_params (dict): Two fixed parameters for the simulation (e.g., {'thickness': 50, 'bulk_n': 1.33}).
        slice_ranges (dict): Ranges for the other two parameters to slice (e.g., {'angle': (60, 70), 'wavelength': (600, 1800)}).
        n_glass (float): Refractive index of the glass substrate. Default is 1.55.
        resolution (int): Number of points to compute across the slice ranges. Default is 100.
        plot (bool): If True, display the resulting spectrum as a plot. Default is True.
        n_layers (list): Optional list of refractive indices for additional intermediate layers. Default is None.
        thickness_layers (list): Optional list of thicknesses for intermediate layers. Must match `n_layers`. Default is None.

    Returns:
        tuple: (X, Y, Z)
            - **X** (ndarray): 1D array of x-values (parameter 1 from `slice_ranges`).
            - **Y** (ndarray): 1D array of y-values (parameter 2 from `slice_ranges`).
            - **Z** (ndarray): 2D array of computed intensity values for the parameter space.

    Example:
        >>> fixed_parameters = {'thickness': 50, 'bulk_n': 1.33}
        >>> slice_ranges = {'angle': (60, 70), 'wavelength': (600, 1800)}
        >>> X, Y, Z = colormesh(material='Au', fixed_params=fixed_parameters, slice_ranges=slice_ranges, resolution=200)
    """

    # Validate input
    if len(fixed_params) != 2 or len(slice_ranges) != 2:
        raise ValueError("You must specify exactly two fixed parameters and two slice ranges.")

    # Extract fixed parameters and ranges
    fixed_keys = list(fixed_params.keys())
    slice_keys = list(slice_ranges.keys())

    # Generate ranges for slicing
    x_range = np.linspace(*slice_ranges[slice_keys[0]], resolution)
    y_range = np.linspace(*slice_ranges[slice_keys[1]], resolution)

    # Create meshgrid for x_range and y_range
    X, Y = np.meshgrid(x_range, y_range)

    # Initialize Z matrix
    Z = np.zeros_like(X, dtype=float)

    # Use tqdm in the loop to track progress for x_range
    for i, x_val in tqdm(enumerate(x_range), total=len(x_range), desc="Computing intensities", ncols=100):
        # Evaluate the intensity for the full Y column for x_val
        Z[:, i] = [
            intensity(material=material, n_glass=n_glass, **{
                slice_keys[0]: x_val,
                slice_keys[1]: y_val,
                fixed_keys[0]: fixed_params[fixed_keys[0]],
                fixed_keys[1]: fixed_params[fixed_keys[1]],
                'n_layers': n_layers,
                'thickness_layers': thickness_layers
            }) for y_val in y_range
        ]

    # Units for plot
    units = {
        'angle': '[°]',
        'wavelength': '[nm]',
        'thickness': '[nm]',
        'bulk_n': '',
    }

    # Plot the results
    if plot:
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(x_range, y_range, Z, shading='auto')
        plt.colorbar(label='Intensity')
        slice_keys = list(slice_ranges.keys())

        # Automatically add units to axis labels
        x_label = f"{slice_keys[0]} {units.get(slice_keys[0], '')}"
        y_label = f"{slice_keys[1]} {units.get(slice_keys[1], '')}"
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        # Construct the title with fixed parameters and material name
        fixed_params_str = ', '.join(
            [f"{key}={value} {units.get(key, '')}" for key, value in fixed_params.items()]
        )
        plt.title(f'SPR Spectrum for {material} (Fixed Parameters: {fixed_params_str})')

        plt.show()

    return x_range, y_range, Z

########################################################################################################################


if __name__ == "__main__":
    # Example usage
    fixed_parameters = {'thickness': 50, 'bulk_n': 1.33}
    slice_ranges = {'angle': (60.0, 70.0), 'wavelength': (600, 1800)}

    X, Y, Z = colormesh(
        material='Ag',
        fixed_params=fixed_parameters,
        slice_ranges=slice_ranges,
        n_glass=1.51,
        resolution=200,
    )

