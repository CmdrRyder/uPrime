"""
core/spatial_spectra_fft.py
----------------------------
3D Spatial spectral analysis using pyfftw for efficient FFT computation.

This module implements 3D spectral analysis following the procedure:
1. Subtract temporal mean <u>(x) from each snapshot -> u'(x,t)
2. Forward transform u'(x,t) into spectral space u_fft(kx, ky, kz, t) using pyfftw
3. Compute energy spectra: (|u_fft|^2 + |v_fft|^2 + |w_fft|^2) * 0.5
4. Bin spectra into wavenumber shells based on K = sqrt(kx^2 + ky^2 + kz^2)
5. Average spectra over time

Reference: FluidFFT (doi:10.5334/jors.238) and FluidSim (10.5334/jors.239) by Mohanan et al.
"""

import numpy as np
import pyfftw

# Configure pyfftw for optimal performance
pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(30.0)


def create_fft_arrays_and_plan(nz, ny, nx):
    """
    Create aligned arrays and a single FFTW plan for efficient 3D FFT computation.

    Parameters
    ----------
    nz, ny, nx : int
                 Dimensions of the spatial grid

    Returns
    -------
    input_array : pyfftw array
                  Aligned input array for FFTW
    output_array : pyfftw array
                   Aligned output array for FFTW
    fft_plan : pyfftw.FFTW
              Single FFTW plan that can be reused
    """
    # Create aligned arrays for pyfftw
    input_array = pyfftw.n_byte_align_empty((nz, ny, nx), 16, dtype="float64")

    # For rfftn, output shape is (nz, ny, nx//2 + 1)
    output_array = pyfftw.n_byte_align_empty(
        (nz, ny, nx // 2 + 1), 16, dtype="complex128"
    )

    # Create a single FFTW plan
    fft_plan = pyfftw.FFTW(
        input_array, output_array, axes=(0, 1, 2), flags=("FFTW_MEASURE",)
    )

    return input_array, output_array, fft_plan


def compute_3d_spectra(U, V, W, Lx, Ly, Lz):
    """
    Compute 3D energy spectra from velocity fields using pyfftw.

    Parameters
    ----------
    U, V, W : 3D arrays of shape (nz, ny, nx, nt)
              Velocity components in x, y, z directions
    Lx, Ly, Lz : float
                 Domain lengths in x, y, z directions (meters)

    Returns
    -------
    k_bins : 1D array
             Wavenumber bin centers [rad/m]
    spectrum_3d : 1D array
                 Time-averaged 3D energy spectrum [m^3/s^2]
    """
    # Get dimensions
    nz, ny, nx, nt = U.shape

    # Compute grid spacing
    dx = Lx / nx
    dy = Ly / ny
    dz = Lz / nz

    # Create wavenumber grids using fftfreq for all directions
    # This gives us the correct symmetric wavenumbers for each dimension
    # kx_full = 2 * np.pi * np.fft.fftfreq(nx, d=dx)  # [rad/m] - shape (nx,)
    ky_full = 2 * np.pi * np.fft.fftfreq(ny, d=dy)  # [rad/m] - shape (ny,)
    kz_full = 2 * np.pi * np.fft.fftfreq(nz, d=dz)  # [rad/m] - shape (nz,)

    # Create meshgrid for full 3D wavenumbers (for proper binning)
    # KX_full, KY_full, KZ_full = np.meshgrid(kx_full, ky_full, kz_full, indexing="ij")
    # K_magnitude_full = np.sqrt(KX_full**2 + KY_full**2 + KZ_full**2)

    # For rfftn output, we only have non-negative frequencies in x-direction
    # So we need to create a wavenumber grid that matches the FFT output shape
    # FFT output shape: (nz, ny, nx//2 + 1)
    # Create proper wavenumber grid for the actual FFT output
    kx_rfft = 2 * np.pi * np.fft.rfftfreq(nx, d=dx)  # [rad/m] - shape (nx//2 + 1,)
    KZ_fft, KY_fft, KX_fft = np.meshgrid(kz_full, ky_full, kx_rfft, indexing="ij")
    K_magnitude = np.sqrt(KX_fft**2 + KY_fft**2 + KZ_fft**2)
    
    # Determine wavenumber bins (logarithmic spacing)
    k_min = np.min(K_magnitude[K_magnitude > 0])
    k_max = np.max(K_magnitude)

    # Calculate wavenumber resolution for each direction
    deltakx = 2 * np.pi / Lx
    deltaky = 2 * np.pi / Ly
    deltakz = 2 * np.pi / Lz

    # Use average wavenumber resolution to determine number of bins
    # This ensures we capture the spectral resolution appropriately
    avg_deltak = (deltakx + deltaky + deltakz) / 3

    # Calculate number of bins based on the wavenumber range and resolution
    # We want enough bins to resolve the spectrum but not too many to be noisy
    n_bins = max(20, min(100, int((k_max - k_min) / avg_deltak)))
    k_bins = np.logspace(np.log10(k_min), np.log10(k_max), n_bins + 1)
    k_bin_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    
    # Initialize spectrum accumulator
    spectrum_accum = np.zeros(n_bins)
    
    # Setup pyfftw for efficient 3D FFTs using a single shared plan
    input_array, output_array, fft_plan = create_fft_arrays_and_plan(nz, ny, nx)
    
    # Process each time step
    for t in range(nt):
        # Step 1: Subtract temporal mean (already done if U, V, W are fluctuations)
        # For this implementation, we assume input are fluctuations u'(x,t)
        u_fluct = U[:, :, :, t]
        v_fluct = V[:, :, :, t]
        w_fluct = W[:, :, :, t]
        
        # Step 2: Forward transform to spectral space using pyfftw with single plan
        # Process u component
        input_array[:] = u_fluct
        fft_plan()
        u_fft = output_array.copy()
        
        # Process v component
        input_array[:] = v_fluct
        fft_plan()
        v_fft = output_array.copy()
        
        # Process w component
        input_array[:] = w_fluct
        fft_plan()
        w_fft = output_array.copy()
        
        # Step 3: Compute energy spectrum
        # E(k) = 0.5 * (|u_fft|^2 + |v_fft|^2 + |w_fft|^2)
        energy_spectrum = 0.5 * (
            np.abs(u_fft) ** 2 + np.abs(v_fft) ** 2 + np.abs(w_fft) ** 2
        )
        
        # Step 4: Bin into wavenumber shells
        # energy_spectrum has shape (nz, ny, nx//2 + 1) due to rfftn
        # K_magnitude should have the same shape
        
        # Flatten arrays for binning
        k_flat = K_magnitude.ravel()
        energy_flat = energy_spectrum.ravel()
        
        # Bin the energy into wavenumber shells
        for i in range(n_bins):
            mask = (k_flat >= k_bins[i]) & (k_flat < k_bins[i + 1])
            if np.any(mask):  # Only sum if there are elements in this bin
                spectrum_accum[i] += np.sum(energy_flat[mask])
    
    # Step 5: Average over time and normalize
    spectrum_3d = spectrum_accum / nt
    
    # Normalize by bin width (for proper spectral density)
    bin_widths = k_bins[1:] - k_bins[:-1]
    spectrum_3d /= bin_widths
    
    return k_bin_centers, spectrum_3d


def compute_1d_spectra(U, V, W, Lx, Ly, Lz):
    """
    Compute 1D energy spectra by integrating over 2D shells.

    Parameters
    ----------
    U, V, W : 3D arrays of shape (nz, ny, nx, nt)
              Velocity components
    Lx, Ly, Lz : float
                 Domain lengths (meters)

    Returns
    -------
    result : dict
             Dictionary containing wavenumbers and spectra for each component
             Keys: 'kx', 'ky', 'kz', 'u_kx', 'u_ky', 'u_kz', 'v_kx', 'v_ky', 'v_kz', 'w_kx', 'w_ky', 'w_kz'
    """
    nz, ny, nx, nt = U.shape
    
    # Compute wavenumbers
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz
    kx = 2 * np.pi * np.fft.rfftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.rfftfreq(ny, d=dy)
    kz = 2 * np.pi * np.fft.rfftfreq(nz, d=dz)
    
    # Initialize accumulators for each component
    spectrum_kx_accum_u = np.zeros(len(kx))
    spectrum_kx_accum_v = np.zeros(len(kx))
    spectrum_kx_accum_w = np.zeros(len(kx))

    spectrum_ky_accum_u = np.zeros(len(ky))
    spectrum_ky_accum_v = np.zeros(len(ky))
    spectrum_ky_accum_w = np.zeros(len(ky))

    spectrum_kz_accum_u = np.zeros(len(kz))
    spectrum_kz_accum_v = np.zeros(len(kz))
    spectrum_kz_accum_w = np.zeros(len(kz))
    
    # Setup pyfftw for efficient 3D FFTs in 1D spectra computation using single plan
    input_array_1d, output_array_1d, fft_plan_1d = create_fft_arrays_and_plan(
        nz, ny, nx
    )

    for t in range(nt):
        # Get fluctuations (assuming input are already fluctuations)
        u_fluct = U[:, :, :, t]
        v_fluct = V[:, :, :, t]
        w_fluct = W[:, :, :, t]
        
        # Compute FFTs using pyfftw with single plan
        # Process u component
        input_array_1d[:] = u_fluct
        fft_plan_1d()
        u_fft = output_array_1d.copy()
        
        # Process v component
        input_array_1d[:] = v_fluct
        fft_plan_1d()
        v_fft = output_array_1d.copy()
        
        # Process w component
        input_array_1d[:] = w_fluct
        fft_plan_1d()
        w_fft = output_array_1d.copy()
        
        # Compute energy spectra for each component separately
        # No sum of components if we are calculating 1D "energy" spectra
        energy_spectrum_u = 0.5 * np.abs(u_fft) ** 2
        energy_spectrum_v = 0.5 * np.abs(v_fft) ** 2
        energy_spectrum_w = 0.5 * np.abs(w_fft) ** 2
        
        # Integrate over y,z planes for kx spectrum
        for i, kx_val in enumerate(kx):
            # For rfftn, kx=0 is at index 0, positive kx at indices 1:
            # and negative kx would be mirrored but rfftn only returns non-negative
            # kx values are in the last dimension (x-axis)
            spectrum_kx_accum_u[i] += np.sum(energy_spectrum_u[:, :, i])
            spectrum_kx_accum_v[i] += np.sum(energy_spectrum_v[:, :, i])
            spectrum_kx_accum_w[i] += np.sum(energy_spectrum_w[:, :, i])
        
        # Integrate over x,z planes for ky spectrum
        for j, ky_val in enumerate(ky):
            if ky_val == 0:
                spectrum_ky_accum_u[j] += np.sum(energy_spectrum_u[:, j, :])
                spectrum_ky_accum_v[j] += np.sum(energy_spectrum_v[:, j, :])
                spectrum_ky_accum_w[j] += np.sum(energy_spectrum_w[:, j, :])
            else:
                # For ky, we need to handle both positive and negative
                # Since FFT is symmetric, we can double the positive part
                spectrum_ky_accum_u[j] += np.sum(energy_spectrum_u[:, j, :])
                spectrum_ky_accum_v[j] += np.sum(energy_spectrum_v[:, j, :])
                spectrum_ky_accum_w[j] += np.sum(energy_spectrum_w[:, j, :])
        
        # Integrate over x,y planes for kz spectrum
        for k, kz_val in enumerate(kz):
            if kz_val == 0:
                spectrum_kz_accum_u[k] += np.sum(energy_spectrum_u[:, :, k])
                spectrum_kz_accum_v[k] += np.sum(energy_spectrum_v[:, :, k])
                spectrum_kz_accum_w[k] += np.sum(energy_spectrum_w[:, :, k])
            else:
                spectrum_kz_accum_u[k] += np.sum(energy_spectrum_u[:, :, k])
                spectrum_kz_accum_v[k] += np.sum(energy_spectrum_v[:, :, k])
                spectrum_kz_accum_w[k] += np.sum(energy_spectrum_w[:, :, k])
    
    # Average over time
    spectrum_kx_u = spectrum_kx_accum_u / nt
    spectrum_kx_v = spectrum_kx_accum_v / nt
    spectrum_kx_w = spectrum_kx_accum_w / nt

    spectrum_ky_u = spectrum_ky_accum_u / nt
    spectrum_ky_v = spectrum_ky_accum_v / nt
    spectrum_ky_w = spectrum_ky_accum_w / nt

    spectrum_kz_u = spectrum_kz_accum_u / nt
    spectrum_kz_v = spectrum_kz_accum_v / nt
    spectrum_kz_w = spectrum_kz_accum_w / nt
    
    # Normalize (account for symmetry in 2D shells)
    # For kx and kz, rfftn already gives us the right normalization
    # For ky, we need to account for both positive and negative wavenumbers
    spectrum_ky_u[1:] *= 2  # Double the energy for positive ky (except DC)
    spectrum_ky_v[1:] *= 2  # Double the energy for positive ky (except DC)
    spectrum_ky_w[1:] *= 2  # Double the energy for positive ky (except DC)

    return {
        "kx": kx,
        "ky": ky,
        "kz": kz,
        "u_kx": spectrum_kx_u,
        "u_ky": spectrum_ky_u,
        "u_kz": spectrum_kz_u,
        "v_kx": spectrum_kx_v,
        "v_ky": spectrum_ky_v,
        "v_kz": spectrum_kz_v,
        "w_kx": spectrum_kx_w,
        "w_ky": spectrum_ky_w,
        "w_kz": spectrum_kz_w,
    }


def compute_spectra_from_fluctuations(U_fluct, V_fluct, W_fluct, Lx, Ly, Lz):
    """
    Compute both 3D and 1D spectra from velocity fluctuations.

    Parameters
    ----------
    U_fluct, V_fluct, W_fluct : 3D arrays of shape (nz, ny, nx, nt)
                                 Velocity fluctuations (temporal mean already subtracted)
    Lx, Ly, Lz : float
                 Domain lengths in meters

    Returns
    -------
    result : dict
             Contains 'k_3d', 'spectrum_3d', 'kx', 'u_kx', 'v_kx', 'w_kx', etc.
    """
    result = {}

    # Get dimensions to create reusable FFT arrays and plans
    nz, ny, nx, nt = U_fluct.shape

    # Compute 3D spectrum
    result["k_3d"], result["spectrum_3d"] = compute_3d_spectra(
        U_fluct, V_fluct, W_fluct, Lx, Ly, Lz
    )

    # Compute 1D spectra
    spectra_1d = compute_1d_spectra(U_fluct, V_fluct, W_fluct, Lx, Ly, Lz)
    result.update(spectra_1d)

    return result


def subtract_temporal_mean(U, V, W):
    """
    Subtract temporal mean from velocity fields to get fluctuations.

    Parameters
    ----------
    U, V, W : 3D arrays of shape (nz, ny, nx, nt)
              Original velocity fields

    Returns
    -------
    U_fluct, V_fluct, W_fluct : 3D arrays
                               Velocity fluctuations u'(x,t) = u(x,t) - <u>(x)
    """
    # Compute temporal means (nanmean tolerates any residual NaNs)
    U_mean = np.nanmean(U, axis=3, keepdims=True)
    V_mean = np.nanmean(V, axis=3, keepdims=True)
    W_mean = np.nanmean(W, axis=3, keepdims=True)
    
    # Subtract to get fluctuations
    U_fluct = U - U_mean
    V_fluct = V - V_mean
    W_fluct = W - W_mean
    
    return U_fluct, V_fluct, W_fluct
