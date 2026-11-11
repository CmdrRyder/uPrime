"""
Analyse a synthetic velocity field with uprime.core, without the GUI.

The field is a uniform mean flow plus one standing mode, so the exact answer
is known in advance:

    u(x, y, t) = U0 + phi(x, y) * a(t)
    a(t) = sqrt(2) * sin(2 * pi * f * t), so that the time mean of a^2 is 1
    <u'u'> = phi^2

The script compares Reynolds stresses and POD with that result.

Run from anywhere after installing uPrime:

    python core_api_example.py
"""

import numpy as np

from uprime.core.pod import compute_pod
from uprime.core.reynolds_stress import compute_reynolds_stresses

# Grid: uPrime stores 2D fields as [ny, nx] and time series as [ny, nx, Nt].
ny, nx, nt = 40, 60, 200
x = np.linspace(0.0, 60.0, nx)  # mm
y = np.linspace(0.0, 40.0, ny)  # mm
X, Y = np.meshgrid(x, y)

# One standing mode, sampled over a whole number of periods.
U0 = 5.0  # m/s
amplitude = 0.5  # m/s
phi = amplitude * np.sin(2.0 * np.pi * X / x[-1]) * np.sin(np.pi * Y / y[-1])
t = np.arange(nt) / nt
a = np.sqrt(2.0) * np.sin(2.0 * np.pi * 5.0 * t)

U = U0 + phi[:, :, np.newaxis] * a
V = np.zeros_like(U)
W = None  # None means planar (2D2C) data

# Reynolds stresses: returns a dict of [ny, nx] fields and the TKE.
stresses, k = compute_reynolds_stresses(U, V, W)
stress_error = np.max(np.abs(stresses["uu"] - phi**2))

# Snapshot POD: the single mode should hold all of the fluctuation energy.
pod = compute_pod(U, V, W, n_modes=5)
first_mode_energy = float(pod["energy_frac"][0])

print(f"Largest error in <u'u'>:     {stress_error:.3e} (m/s)^2")
print(f"Energy in the first POD mode: {100.0 * first_mode_energy:.4f} %")

assert stress_error < 1e-10
assert first_mode_energy > 0.9999
