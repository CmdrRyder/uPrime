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
    kx_full = 2 * np.pi * np.fft.fftfreq(nx, d=dx)  # [rad/m] - shape (nx,)
    ky_full = 2 * np.pi * np.fft.fftfreq(ny, d=dy)  # [rad/m] - shape (ny,)
    kz_full = 2 * np.pi * np.fft.fftfreq(nz, d=dz)  # [rad/m] - shape (nz,)
    
    # Create meshgrid for full 3D wavenumbers (for proper binning)
    KX_full, KY_full, KZ_full = np.meshgrid(kx_full, ky_full, kz_full, indexing='ij')
    K_magnitude_full = np.sqrt(KX_full**2 + KY_full**2 + KZ_full**2)
    
    # For rfftn output, we only have non-negative frequencies in x-direction
    # So we need to create a wavenumber grid that matches the FFT output shape
    # FFT output shape: (nz, ny, nx//2 + 1)
    # Create proper wavenumber grid for the actual FFT output
    kx_rfft = 2 * np.pi * np.fft.rfftfreq(nx, d=dx)  # [rad/m] - shape (nx//2 + 1,)
    KZ_fft, KY_fft, KX_fft = np.meshgrid(kz_full, ky_full, kx_rfft, indexing='ij')
    K_magnitude = np.sqrt(KX_fft**2 + KY_fft**2 + KZ_fft**2)
    
    # Determine wavenumber bins (logarithmic spacing)
    k_min = np.min(K_magnitude[K_magnitude > 0])
    k_max = np.max(K_magnitude)
    n_bins = 50
    k_bins = np.logspace(np.log10(k_min), np.log10(k_max), n_bins + 1)
    k_bin_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    
    # Initialize spectrum accumulator
    spectrum_accum = np.zeros(n_bins)
    
    # Setup pyfftw for efficient 3D FFTs
    # Create aligned arrays for pyfftw
    u_fluct_aligned = pyfftw.n_byte_align_empty((nz, ny, nx), 16, dtype='float64')
    v_fluct_aligned = pyfftw.n_byte_align_empty((nz, ny, nx), 16, dtype='float64')
    w_fluct_aligned = pyfftw.n_byte_align_empty((nz, ny, nx), 16, dtype='float64')
    
    # For rfftn, output shape is (nz, ny, nx//2 + 1)
    u_fft_aligned = pyfftw.n_byte_align_empty((nz, ny, nx//2 + 1), 16, dtype='complex128')
    v_fft_aligned = pyfftw.n_byte_align_empty((nz, ny, nx//2 + 1), 16, dtype='complex128')
    w_fft_aligned = pyfftw.n_byte_align_empty((nz, ny, nx//2 + 1), 16, dtype='complex128')
    
    # Create FFTW plans
    fft_u = pyfftw.FFTW(u_fluct_aligned, u_fft_aligned, axes=(0, 1, 2), flags=('FFTW_MEASURE',))
    fft_v = pyfftw.FFTW(v_fluct_aligned, v_fft_aligned, axes=(0, 1, 2), flags=('FFTW_MEASURE',))
    fft_w = pyfftw.FFTW(w_fluct_aligned, w_fft_aligned, axes=(0, 1, 2), flags=('FFTW_MEASURE',))
    
    # Process each time step
    for t in range(nt):
        # Step 1: Subtract temporal mean (already done if U, V, W are fluctuations)
        # For this implementation, we assume input are fluctuations u'(x,t)
        u_fluct = U[:, :, :, t]
        v_fluct = V[:, :, :, t]
        w_fluct = W[:, :, :, t]
        
        # Copy data to aligned arrays
        u_fluct_aligned[:] = u_fluct
        v_fluct_aligned[:] = v_fluct
        w_fluct_aligned[:] = w_fluct
        
        # Step 2: Forward transform to spectral space using pyfftw
        fft_u()
        fft_v()
        fft_w()
        
        # Get the FFT results
        u_fft = u_fft_aligned.copy()
        v_fft = v_fft_aligned.copy()
        w_fft = w_fft_aligned.copy()
        
        # Step 3: Compute energy spectrum
        # E(k) = 0.5 * (|u_fft|^2 + |v_fft|^2 + |w_fft|^2)
        energy_spectrum = 0.5 * (np.abs(u_fft)**2 + np.abs(v_fft)**2 + np.abs(w_fft)**2)
        
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
    kx_1d : 1D array
            Wavenumber in x-direction [rad/m]
    spectrum_kx : 1D array
                 Energy spectrum vs kx [m^3/s^2]
    ky_1d : 1D array
            Wavenumber in y-direction [rad/m]
    spectrum_ky : 1D array
                 Energy spectrum vs ky [m^3/s^2]
    kz_1d : 1D array
            Wavenumber in z-direction [rad/m]
    spectrum_kz : 1D array
                 Energy spectrum vs kz [m^3/s^2]
    """
    nz, ny, nx, nt = U.shape
    
    # Compute wavenumbers
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz
    kx = 2 * np.pi * np.fft.rfftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.rfftfreq(ny, d=dy)
    kz = 2 * np.pi * np.fft.rfftfreq(nz, d=dz)
    
    # Initialize accumulators
    spectrum_kx_accum = np.zeros(len(kx))
    spectrum_ky_accum = np.zeros(len(ky))
    spectrum_kz_accum = np.zeros(len(kz))
    
    # Setup pyfftw for efficient 3D FFTs in 1D spectra computation
    u_fluct_aligned_1d = pyfftw.n_byte_align_empty((nz, ny, nx), 16, dtype='float64')
    v_fluct_aligned_1d = pyfftw.n_byte_align_empty((nz, ny, nx), 16, dtype='float64')
    w_fluct_aligned_1d = pyfftw.n_byte_align_empty((nz, ny, nx), 16, dtype='float64')
    
    u_fft_aligned_1d = pyfftw.n_byte_align_empty((nz, ny, nx//2 + 1), 16, dtype='complex128')
    v_fft_aligned_1d = pyfftw.n_byte_align_empty((nz, ny, nx//2 + 1), 16, dtype='complex128')
    w_fft_aligned_1d = pyfftw.n_byte_align_empty((nz, ny, nx//2 + 1), 16, dtype='complex128')
    
    fft_u_1d = pyfftw.FFTW(u_fluct_aligned_1d, u_fft_aligned_1d, axes=(0, 1, 2), flags=('FFTW_MEASURE',))
    fft_v_1d = pyfftw.FFTW(v_fluct_aligned_1d, v_fft_aligned_1d, axes=(0, 1, 2), flags=('FFTW_MEASURE',))
    fft_w_1d = pyfftw.FFTW(w_fluct_aligned_1d, w_fft_aligned_1d, axes=(0, 1, 2), flags=('FFTW_MEASURE',))
    
    for t in range(nt):
        # Get fluctuations (assuming input are already fluctuations)
        u_fluct = U[:, :, :, t]
        v_fluct = V[:, :, :, t]
        w_fluct = W[:, :, :, t]
        
        # Copy to aligned arrays
        u_fluct_aligned_1d[:] = u_fluct
        v_fluct_aligned_1d[:] = v_fluct
        w_fluct_aligned_1d[:] = w_fluct
        
        # Compute FFTs using pyfftw
        fft_u_1d()
        fft_v_1d()
        fft_w_1d()
        
        # Get results
        u_fft = u_fft_aligned_1d.copy()
        v_fft = v_fft_aligned_1d.copy()
        w_fft = w_fft_aligned_1d.copy()
        
        # Compute energy spectrum
        energy_spectrum = 0.5 * (np.abs(u_fft)**2 + np.abs(v_fft)**2 + np.abs(w_fft)**2)
        
        # Integrate over y,z planes for kx spectrum
        for i, kx_val in enumerate(kx):
            # For rfftn, kx=0 is at index 0, positive kx at indices 1:
            # and negative kx would be mirrored but rfftn only returns non-negative
            if i == 0:
                # kx=0 case
                spectrum_kx_accum[i] += np.sum(energy_spectrum[0, :, :])
            else:
                # Positive kx values
                spectrum_kx_accum[i] += np.sum(energy_spectrum[i, :, :])
        
        # Integrate over x,z planes for ky spectrum
        for j, ky_val in enumerate(ky):
            if ky_val == 0:
                spectrum_ky_accum[j] += np.sum(energy_spectrum[:, j, :])
            else:
                # For ky, we need to handle both positive and negative
                # Since FFT is symmetric, we can double the positive part
                spectrum_ky_accum[j] += np.sum(energy_spectrum[:, j, :])
        
        # Integrate over x,y planes for kz spectrum
        for k, kz_val in enumerate(kz):
            if kz_val == 0:
                spectrum_kz_accum[k] += np.sum(energy_spectrum[:, :, k])
            else:
                spectrum_kz_accum[k] += np.sum(energy_spectrum[:, :, k])
    
    # Average over time
    spectrum_kx = spectrum_kx_accum / nt
    spectrum_ky = spectrum_ky_accum / nt
    spectrum_kz = spectrum_kz_accum / nt
    
    # Normalize (account for symmetry in 2D shells)
    # For kx and kz, rfftn already gives us the right normalization
    # For ky, we need to account for both positive and negative wavenumbers
    spectrum_ky[1:] *= 2  # Double the energy for positive ky (except DC)
    
    return kx, spectrum_kx, ky, spectrum_ky, kz, spectrum_kz


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
             Contains 'k_3d', 'spectrum_3d', 'kx', 'spectrum_kx', etc.
    """
    result = {}
    
    # Compute 3D spectrum
    result['k_3d'], result['spectrum_3d'] = compute_3d_spectra(U_fluct, V_fluct, W_fluct, Lx, Ly, Lz)
    
    # Compute 1D spectra
    result['kx'], result['spectrum_kx'], \
    result['ky'], result['spectrum_ky'], \
    result['kz'], result['spectrum_kz'] = compute_1d_spectra(U_fluct, V_fluct, W_fluct, Lx, Ly, Lz)
    
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
    # Compute temporal means
    U_mean = np.mean(U, axis=3, keepdims=True)
    V_mean = np.mean(V, axis=3, keepdims=True)
    W_mean = np.mean(W, axis=3, keepdims=True)
    
    # Subtract to get fluctuations
    U_fluct = U - U_mean
    V_fluct = V - V_mean
    W_fluct = W - W_mean
    
    return U_fluct, V_fluct, W_fluct
