import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from colormesh import colormesh

# Example usage

# Fix z and w, slice across x and y
fixed_parameters = {'thickness': 50, 'RI': 1.33}
slice_ranges = {'angle': (58.0, 67.0), 'wavelength': (500, 1800)}

X_1, Y_1, Z_1 = colormesh(material='Ag',
                        fixed_params=fixed_parameters,
                        slice_ranges=slice_ranges,
                        resolution=200,
                        plot=False
                    )

# Layer of higher index of refraction
n_layer = [1.35]
thickness_layers = [100]

X_2, Y_2, Z_2 = colormesh(material='Ag',
                        fixed_params=fixed_parameters,
                        slice_ranges=slice_ranges,
                        resolution=200,
                        n_layers=n_layer,
                        thickness_layers=thickness_layers,
                        plot=False
                    )

# Assuming X_2, Y_2, Z_2 and X_1, Y_1, Z_1 are already defined
# Subtract the Z values of Z_2 and Z_1
Z_diff = Z_2 - Z_1

# Plot the difference
plt.figure(figsize=(8, 6))
plt.contourf(X_2, Y_2, Z_diff, cmap="coolwarm")
plt.colorbar(label="Difference in Z")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Difference in Z values (Z_2 - Z_1)")
plt.grid(True)
plt.show()
