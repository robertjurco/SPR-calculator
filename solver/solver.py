import numpy as np
from functools import lru_cache
import matplotlib.pyplot as plt

from materials.material import material_n

def gamma(n, L, angle, polarity):
    N = len(n) - 2  # number of slabs (assuming n is stacked with a first and last layer as glass and bulk)
    angle = angle * np.pi / 180.0  # Convert angle to radians

    # Initialize costh, which is the cosine of the angle (can be complex)
    costh = np.zeros_like(n, dtype=complex)
    for i in range(len(n)):
        costh[i] = 1 - (n[0] * np.sin(angle) / n[i]) ** 2
        # Ensure complex square roots are handled correctly
        condition = np.real(costh[i]) >= 0
        costh[i] = np.where(condition, np.sqrt(costh[i]), np.conj(np.sqrt(costh[i])))

    # Depending on the polarity (TE or TM), we compute nT
    if polarity == 'TE':
        nT = n * costh  # Transverse Electric case
    else:
        nT = n / costh  # Transverse Magnetic case

    # Adjust the thickness layers according to the cos(angle) for each layer
    for i in range(N):
        L[i] *= costh[i + 1]  # n(i) * L(i) * cos(theta(i))

    # Reflection coefficients for each interface
    # Initialize reflection coefficients (element-wise)
    r = np.zeros((N + 1, *nT.shape[1:]), dtype=complex)  # Broadcasting across the 4D shape
    for i in range(N + 1):
        r[i] = np.divide(nT[i] - nT[i + 1], nT[i] + nT[i + 1])  # Element-wise division

    # Initialize Gamma at the right-most interface (bulk layer)
    Gamma = r[N]

    # Iterate over layers in reverse order (starting from the right-most layer)
    for i in range(N - 1, -1, -1):
        delta = 2 * np.pi * L[i]  # phase thickness in i-th layer
        z = np.exp(-2j * delta)  # phase factor
        Gamma = (r[i] + Gamma * z) / (1 + r[i] * Gamma * z)  # update Gamma

    return Gamma


def intensity(angle, wavelength, thickness, bulk_n,
              n_layers: list = None,
              thickness_layers: list = None,
              material='Au',
              n_glass=1.51):
    """
    Calculate the intensity of light for specified angles, wavelengths, and material properties
    in a multilayer optical system.

    Args:
        angle (float or array-like): Angle(s) of incidence in degrees.
        wavelength (float or array-like): Wavelength(s) (in nm) for intensity calculation.
        thickness (float or array-like): Thickness (in nm) of the main layer.
        bulk_n (float or array-like): Refractive index of the bulk or buffer layer.
        n_layers (list of floats, optional): Refractive indices of additional layers.
        thickness_layers (list of floats, optional): Thicknesses (in nm) of additional layers.
        material (str): Material of the main layer (default: 'Au').
        n_glass (float): Refractive index of the substrate (default: 1.51).

    Returns:
        float or numpy.ndarray: Calculated light intensity as a scalar or array.

    Raises:
        ValueError: If n_layers and thickness_layers lengths do not match.

    Notes:
    - Supports both single-value and multi-dimensional inputs via broadcasting.
    - Uses transfer-matrix methods to compute intensities based on optical paths.
    - Additional layers are optional but must have matching refractive index and thickness lists.
    - **Warning**: If the result is used in a colormesh or contour plot with two varying
      parameters (e.g., `angle` and `wavelength`), ensure that the result's slicing order
      matches the axes' order. You may need to transpose the result (`.T`) if the order differs.


    Example:
        >>> intensity(45.0, 632.8, 100.0, 1.4)
        0.942

        >>> intensity(45.0, [500, 600], 50.0, 1.2, n_layers=[1.8], thickness_layers=[10])
        array([0.812, 0.864])
    """

    # Ensure inputs are arrays
    # Reshape inputs for broadcasting
    # Each parameter varies along its respective axis
    angle = np.atleast_1d(angle)[:, None, None, None]       # shape (A,1,1,1)
    wavelength = np.atleast_1d(wavelength)[None, :, None, None]  # shape (1,W,1,1)
    thickness = np.atleast_1d(thickness)[None, None, :, None]    # shape (1,1,T,1)
    bulk_n = np.atleast_1d(bulk_n)[None, None, None, :]     # shape (1,1,1,B)

    shape = angle.shape[0], wavelength.shape[1], thickness.shape[2], bulk_n.shape[3]

    # If additional layers are provided, check them
    # Default empty lists if None
    n_layers = n_layers or []
    thickness_layers = thickness_layers or []

    # Check lengths if both are provided
    if n_layers or thickness_layers:
        if len(n_layers) != len(thickness_layers):
            raise ValueError(
                f"Mismatch: n_layers (len={len(n_layers)}) and thickness_layers (len={len(thickness_layers)}) must have the same lengths.")

    # Pre-compute refractive indices
    thickness_ti = 1.5
    n_mat_raw = np.conjugate(material_n(material, wavelength))
    n_ti_raw = np.conjugate(material_n('Ti', wavelength))

    # Broadcast both to (A, W, T, B)
    n_ti = np.broadcast_to(n_ti_raw, shape)
    n_mat = np.broadcast_to(n_mat_raw, shape)

    # Expand scalar n_glass
    n_glass_array = np.full(shape, n_glass, dtype=complex)

    # bulk_n is likely (1,1,1,B), broadcast it too
    bulk_n = np.broadcast_to(bulk_n, shape)

    # Now everything is aligned
    n_stack = [n_glass_array, n_ti, n_mat]

    # Add optional layers
    for n_layer in n_layers or []:
        n_arr = np.array(n_layer)
        if n_arr.ndim == 0:
            n_arr = np.full(shape, n_arr, dtype=complex)
        elif n_arr.shape != shape:
            n_arr = np.broadcast_to(n_arr, shape)
        n_stack.append(n_arr)

    # Add bulk
    n_stack.append(bulk_n)

    # Final stack shape: (N_layers, A, W, T, B)
    n = np.stack(n_stack, axis=0)

    # Precompute thickness values
    thickness_ti = 1.5  # Example for titanium thickness
    L_stack = [thickness_ti * n_ti / wavelength]  # First layer (Ti)

    # For the main layer, we compute L from the main thickness and refractive index
    L_main = thickness * n_mat / wavelength  # shape (A, W, T, B)
    L_stack.append(L_main)

    # For each additional layer (if provided)
    for i, thickness_layer in enumerate(thickness_layers or []):
        thickness_layer = np.array(thickness_layer)
        if thickness_layer.ndim == 0:
            # If it's a scalar, broadcast it to full shape
            L_layer = np.full(shape, thickness_layer * n_layers[i] / wavelength, dtype=complex)
        elif thickness_layer.shape != shape:
            # If it's an array, broadcast it to the correct shape
            L_layer = np.broadcast_to(thickness_layer, shape) * n_layers[i] / wavelength
        L_stack.append(L_layer)

    # For bulk layer (final layer)
    L_bulk = np.full(shape, 1.0 * bulk_n / wavelength, dtype=complex)  # Assuming the bulk layer is 1 unit thick
    L_stack.append(L_bulk)

    # Stack all L values together (N_layers, A, W, T, B)
    L = np.stack(L_stack, axis=0)

    # Calculate gamma and the result
    gamma_return = gamma(n, L, angle, 'tm')
    result = np.abs(gamma_return) ** 2

    # Transpose to keep correct order
    return np.squeeze(result).T


if __name__ == "__main__":
    def main():
        # Parameter ranges
        angle_range = np.linspace(60, 70, 500)  # 500 points between 60° and 70°
        wavelength_range = np.linspace(500, 2000, 500)  # 500 points between 500nm and 2000nm
        thickness_range = np.linspace(10, 100, 500)  # 500 points for thickness between 10 nm and 100 nm
        bulk_n_range = np.linspace(1.2, 1.37, 500)  # 500 points for bulk refractive index

        # Fixed parameters
        material = 'Au'  # Example material
        thickness_fixed = 50  # Fixed thickness in nm for some views
        bulk_n_fixed = 1.33  # Fixed refractive index of the buffer layer
        angle_fixed = 65  # Fixed angle in degrees

        # 1. Intensity Map: (Angle, Wavelength)
        intensities_angle_wavelength = intensity(
            angle=angle_range,
            wavelength=wavelength_range,
            thickness=thickness_fixed,
            bulk_n=bulk_n_fixed,
            material=material
        )
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(angle_range, wavelength_range, intensities_angle_wavelength, shading='auto', cmap='viridis')
        plt.colorbar(label="Intensity")
        plt.ylabel("Wavelength [nm]")
        plt.xlabel("Angle [°]")
        plt.title("Intensity Map (Angle vs. Wavelength)")
        plt.show()

        # 2. Intensity Map: (Wavelength, Thickness)
        intensities_wavelength_thickness = intensity(
            angle=angle_fixed,
            wavelength=wavelength_range,
            thickness=thickness_range,
            bulk_n=bulk_n_fixed,
            material=material
        )
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(thickness_range, wavelength_range, intensities_wavelength_thickness.T, shading='auto', cmap='viridis')
        plt.colorbar(label="Intensity")
        plt.ylabel("Wavelength [nm]")
        plt.xlabel("Thickness [nm]")
        plt.title("Intensity Map (Wavelength vs. Thickness)")
        plt.show()

        # 3. Intensity Map: (Wavelength, Bulk_n)
        intensities_wavelength_bulk_n = intensity(
            angle=angle_fixed,
            wavelength=wavelength_range,
            thickness=thickness_fixed,
            bulk_n=bulk_n_range,
            material=material
        )
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(bulk_n_range, wavelength_range, intensities_wavelength_bulk_n.T, shading='auto', cmap='viridis')
        plt.colorbar(label="Intensity")
        plt.ylabel("Wavelength [nm]")
        plt.xlabel("Bulk Refractive Index (n)")
        plt.title("Intensity Map (Wavelength vs. Bulk Refractive Index)")
        plt.show()

    main()