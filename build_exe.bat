@echo off
REM ============================================================
REM  build_exe.bat -- uPrime v0.6.1
REM  Builds a single-file windowed executable via PyInstaller
REM  Target: Python 3.11
REM  Usage:  double-click or run from project root
REM
REM  Packages bundled:
REM    matplotlib  -- collect-all (backends, fonts, style data)
REM    PyQt6       -- collect-all (platform plugins, sip binding)
REM    numpy       -- collect-all (compiled extensions)
REM    scipy       -- collect-all (signal, ndimage, interpolate, io.matlab)
REM    pyfftw      -- collect-all + collect-binaries (FFTW DLLs)
REM    h5py        -- collect-all (v7.3 .mat file reading)
REM    PIL/Pillow  -- collect-all (image handling)
REM
REM  New in v0.6.1:
REM    Bugfix release on top of v0.6.0.
REM    - get_masked() preserves input dtype (no longer promotes to float64)
REM    - .mat loader fills masked regions with NaN (not 0) for display
REM    - contourf wrapped with np.ma.masked_invalid so NaN renders as
REM      axes background (white) instead of colormap minimum
REM    - explicit "Mask convention" dropdown in .mat variable dialog
REM      replaces the brittle auto-invert heuristic
REM
REM  Carried from v0.6.0:
REM    MATLAB .mat file input (classic v5/v7 via scipy.io.matlab,
REM      v7.3 HDF5 via h5py); variable confirmation dialog with
REM      snapshot subset spinboxes; opt-in float16 storage for
REM      files >5 GB; TKE Budget stacked-panel line plot
REM
REM  Carried from v0.5.1:
REM    Mean & Convergence module (first tab in main window)
REM      Tab 1: mean velocity line plots (free / horizontal / vertical)
REM      Tab 2: cumulative convergence with Welford online algorithm,
REM             sustained-crossing N* detection
REM
REM  Carried from v0.5.0:
REM    PyQt6.QtCore.QThread -- background workers
REM    tempfile, mmap       -- memory-mapped large dataset support
REM    traceback            -- worker error reporting
REM    concurrent.futures   -- thread pool support
REM ============================================================
python -m pip install pyinstaller --quiet
pyinstaller ^
    --onefile ^
    --windowed ^
    --name uPrime_v0.6.1 ^
    --add-data "assets;assets" ^
    --collect-all matplotlib ^
    --hidden-import matplotlib.backends.backend_qtagg ^
    --hidden-import matplotlib.backends.backend_qt ^
    --hidden-import matplotlib.figure ^
    --hidden-import matplotlib.patches ^
    --hidden-import matplotlib.lines ^
    --hidden-import matplotlib.pyplot ^
    --collect-all PyQt6 ^
    --hidden-import PyQt6.QtWidgets ^
    --hidden-import PyQt6.QtCore ^
    --hidden-import PyQt6.QtGui ^
    --hidden-import PyQt6.QtSvg ^
    --hidden-import PyQt6.sip ^
    --collect-all numpy ^
    --hidden-import numpy.linalg ^
    --hidden-import numpy.lib.stride_tricks ^
    --hidden-import numpy.ma ^
    --collect-all scipy ^
    --hidden-import scipy.signal ^
    --hidden-import scipy.signal.windows ^
    --hidden-import scipy.ndimage ^
    --hidden-import scipy.ndimage.filters ^
    --hidden-import scipy.interpolate ^
    --hidden-import scipy.interpolate.fitpack2 ^
    --hidden-import scipy.linalg ^
    --hidden-import scipy.linalg.blas ^
    --hidden-import scipy.linalg.lapack ^
    --hidden-import scipy.sparse ^
    --hidden-import scipy.sparse.linalg ^
    --hidden-import scipy._lib ^
    --hidden-import scipy._lib.messagestream ^
    --hidden-import scipy.special ^
    --hidden-import scipy.io ^
    --hidden-import scipy.io.matlab ^
    --collect-all pyfftw ^
    --collect-binaries pyfftw ^
    --hidden-import pyfftw ^
    --hidden-import pyfftw.interfaces ^
    --hidden-import pyfftw.interfaces.numpy_fft ^
    --hidden-import pyfftw.interfaces.scipy_fft ^
    --hidden-import pyfftw.interfaces.cache ^
    --collect-all h5py ^
    --hidden-import h5py ^
    --hidden-import h5py.defs ^
    --hidden-import h5py.utils ^
    --hidden-import h5py.h5ac ^
    --hidden-import h5py._proxy ^
    --collect-all PIL ^
    --hidden-import PIL.Image ^
    --hidden-import PIL.ImageDraw ^
    --hidden-import concurrent.futures ^
    --hidden-import tempfile ^
    --hidden-import mmap ^
    --hidden-import csv ^
    --hidden-import datetime ^
    --hidden-import re ^
    --hidden-import traceback ^
    --hidden-import threading ^
    --hidden-import queue ^
    --hidden-import gc ^
    main.py
echo.
if errorlevel 1 (
    echo BUILD FAILED. Check output above.
) else (
    echo BUILD COMPLETE: dist\uPrime_v0.6.1.exe
)
pause