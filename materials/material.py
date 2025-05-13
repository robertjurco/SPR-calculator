import os

import numpy as np
from scipy.interpolate import CubicSpline

################################################## Data ################################################################
# Material data: files or Drude-Lorentz parameters
materials_dict_files = [
    {'material': 'Au', 'file': '../data/gold_yakubovsky.txt'},
    {'material': 'Ag', 'file': '../data/Ag.txt'},
    {'material': 'Cu', 'file': '../data/copper.txt'},
    {'material': 'Al', 'file': '../data/Al.txt'},
    {'material': 'Rh', 'file': '../data/Rhodium.txt'},
    {'material': 'Ti', 'file': '../data/Ti.txt'},
    {'material': 'TiN', 'file': '../data/TiN.txt'}
]

materials_dict_DL = [
    {'material': 'ZrN', 'params': [3.4656, 8.018, 0.5192, 2.4509, 5.48, 1.7369]}
]

############################################## TEST MATERIAL ###########################################################

def test_material(wavelength):
    """Calculate test permittivity for a given wavelength."""
    wavelength_start, wavelength_end = 500, 1700
    eps_real_start, eps_real_end = -0.1, -0.1
    eps_imag_start, eps_imag_end = 0, 1

    eps_real = (eps_real_end - eps_real_start) / (wavelength_end - wavelength_start) * (
                wavelength - wavelength_start) + eps_real_start
    eps_imag = (eps_imag_end - eps_imag_start) / (wavelength_end - wavelength_start) * (
                wavelength - wavelength_start) + eps_imag_start

    return eps_real + 1j * eps_imag


################################################# HELPER ###############################################################

def get_drude_lorentz_params(material: str) -> list[float] | None:
    """
    Retrieve Drude-Lorentz parameters for a given material.

    Args:
        material (str): Material name.

    Returns:
        list[float] | None: Drude-Lorentz parameters if the material exists, otherwise None.
    """
    for entry in materials_dict_DL:
        if entry['material'] == material:
            return entry['params']
    return None


def eps_DrudeLorentz(wavelength, params):
    """
    Calculate permittivity using the Drude-Lorentz model.

    Args:
        wavelength (float): Wavelength in microns.
        params (list[float]): Drude-Lorentz parameters [eps_b, omega_p, gamma_p, f_1, omega_1, gamma_1].

    Returns:
        complex: Permittivity for the given wavelength.
    """
    c = 0.299792458  # Speed of light in µm/fs
    eps_b, omega_p, gamma_p, f_1, omega_1, gamma_1 = params
    omega = 2 * np.pi * c / wavelength

    eps = eps_b - omega_p**2 / omega / (omega + 1.0j * gamma_p) + f_1 * omega_1**2 / (gamma_1**2 - omega**2 - 1.0j*omega*gamma_1)

    return eps


def interpolate_n_from_file(material: str, wavelength: float) -> complex:
    """
    Interpolate refractive index (n) from a file for a specific material and convert it to permittivity (n^2).

    Args:
        material (str): Material name (e.g., 'Au' or 'Ag').
        wavelength (float): Wavelength in microns.

    Returns:
        complex: Permittivity (n^2) for the given wavelength, or None if the file cannot be read.
    """
    for entry in materials_dict_files:
        if entry['material'] == material:
            # Resolve file path relative to this script's directory
            file_path = os.path.join(os.path.dirname(__file__), entry['file'])
            try:
                data = np.loadtxt(file_path)  # Assumes file format: wavelength(nm), n_real, n_imag
                wavelength_data, n_real, n_imag = data.T

                # Perform cubic spline interpolation
                spline_real = CubicSpline(wavelength_data, n_real)
                spline_imag = CubicSpline(wavelength_data, n_imag)

                n = spline_real(wavelength) + 1j * spline_imag(wavelength)
                return n
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                return None
    return None


def get_material_eps(material: str, wavelength: float) -> complex | None:
    """
    Compute permittivity for a single wavelength based on material type.

    Args:
        material (str): Material name (e.g., 'Au' or 'ZrN').
        wavelength (float): Wavelength in microns.

    Returns:
        complex | None: Permittivity for the given material and wavelength.
    """
    # Compute using refractive index data (n^2)
    n = interpolate_n_from_file(material, wavelength)
    if n is not None:
        return n ** 2

    # Compute using Drude-Lorentz model
    params = get_drude_lorentz_params(material)
    if params:
        return eps_DrudeLorentz(wavelength, params)

    # Compute for test material
    if material == 'test':
        return test_material(wavelength)

    # Material not found
    print(f"Material {material} not found.")
    return None


def get_material_n(material: str, wavelength: float) -> complex | None:
    """
    Compute refractive index for a single wavelength based on material type.

    Args:
        material (str): Material name (e.g., 'Au' or 'ZrN').
        wavelength (float): Wavelength in microns.

    Returns:
        complex | None: Refractive index for the given material and wavelength.
    """
    # Compute directly from refractive index data
    n = interpolate_n_from_file(material, wavelength)
    if n is not None:
        return n

    # Compute from sqrt(permittivity) using Drude-Lorentz
    params = get_drude_lorentz_params(material)
    if params:
        eps = eps_DrudeLorentz(wavelength, params)
        if eps is not None:
            return np.sqrt(eps)

    # Compute for test material
    if material == 'test':
        eps = test_material(wavelength)
        return np.sqrt(eps)

    # Material not found
    print(f"Material {material} not found.")
    return None


################################################## MAIN ################################################################

def material_eps(material: str, wavelengths: float | list[float]) -> np.ndarray | complex | None:
    """
    Get permittivity for a material at a given wavelength or list of wavelengths.

    Args:
        material (str): Material name.
        wavelengths (float | list[float]): Single wavelength or list of wavelengths in nm.

    Returns:
        np.ndarray | complex | None: Array of permittivities or a single value, or None if an error occurred.
    """
    if np.isscalar(wavelengths):  # Single wavelength
        return get_material_eps(material, wavelengths / 1000)  # Convert nm to µm

    # Handle iterable wavelengths
    return np.array([get_material_eps(material, wl / 1000) for wl in wavelengths])


def material_n(material: str, wavelengths: float | list[float]) -> np.ndarray | complex | None:
    """
    Get refractive index for a material at a given wavelength or list of wavelengths.

    Args:
        material (str): Material name.
        wavelengths (float | list[float]): Single wavelength or list of wavelengths in nm.

    Returns:
        np.ndarray | complex | None: Array of refractive indices or a single value, or None if an error occurred.
    """
    if np.isscalar(wavelengths):  # Single wavelength
        return get_material_n(material, wavelengths / 1000)  # Convert nm to µm

    # Handle iterable wavelengths
    return np.array([get_material_n(material, wl / 1000) for wl in wavelengths])
