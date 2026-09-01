@echo off
REM ============================================================
REM  build_exe.bat -- uPrime v0.8.0  (PUBLIC build)
REM  Builds a single-file windowed executable via PyInstaller
REM  Target: Python 3.11
REM  Usage:  double-click or run from project root
REM
REM  VARIANT: PUBLIC.  DaVis support is NOT included -- lvpyio is
REM  explicitly excluded so it can never be swept in, even if it
REM  happens to be installed in the build environment.  To use DaVis
REM  (.vc7/.vec) loading, run uPrime from source with lvpyio
REM  installed.
REM
REM  Packages bundled:
REM    matplotlib  -- collect-all (backends, fonts, style data)
REM    PyQt6       -- collect-all (platform plugins, sip binding)
REM    numpy       -- collect-all (compiled extensions)
REM    scipy       -- collect-all (signal, ndimage, interpolate, io.matlab)
REM    pyfftw      -- collect-all + collect-binaries (FFTW DLLs)
REM    h5py        -- collect-all (v7.3 .mat file reading)
REM    PIL/Pillow  -- collect-all (image handling)
REM    openpyxl    -- hidden-import (optional .xlsx reading in Compare
REM                   Cases; harmless warning if openpyxl is absent)
REM
REM  New in v0.8.0 (public build):
REM    - Probability Analysis module (PDF, Flow Direction, Binary Space-Time,
REM      Quadrant tabs). Chunked, memmap-safe, NaN-correct accumulators.
REM
REM  Carried from v0.7.1:
REM    - lvpyio EXCLUDED. LaVision DaVis (.vc7/.vec) loading is unavailable
REM      in this exe. To use DaVis input, run from source with lvpyio
REM      installed.
REM    - Orphaned memmap temp-file sweep at startup and before each load.
REM    - Main window opens ~10% larger (no sidebar scrollbar).
REM
REM  New in v0.7.0:
REM    - Compare Cases viewer (standalone, dataset-independent):
REM        1D Compare tab -- overlay/compare uPrime's own 1D exports
REM          (.csv/.xlsx) grouped by variable; styling, merged-CSV +
REM          300 DPI PNG/PDF/SVG export.
REM        Field Compare (2D) tab -- tiled contour comparison of 2D
REM          .dat fields with shared/per-tile colour scaling, plus a
REM          line-extract mode that samples all components along a
REM          drawn/manual line into the 1D basket.
REM    - Manual start/end coordinates for line profiles (Reynolds,
REM      TKE Budget, Mean Velocity), alongside drawn lines.
REM    - TKE normalization fix: k/Um^2 (upper) and (P L)/Um^3 (lower)
REM      applied per panel; normalize checkbox relabelled.
REM    - Unified 2D .dat export (single canonical Tecplot writer) and
REM      case-name-prefixed default filenames for .dat exports.
REM    - Optional DaVis (.vc7/.vec) vector-file support via lvpyio
REM      (EXCLUDED from this public build).
REM    - Neutral-grey control indicators; version read from version.txt.
REM
REM  Carried from v0.6.x:
REM    MATLAB .mat input (classic via scipy.io.matlab, v7.3 via h5py);
REM    variable dialog with snapshot subset + mask convention; opt-in
REM    float16 for files >5 GB; memory-mapped large datasets; TKE
REM    Budget stacked-panel line plot; Mean & Convergence module.
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
    --name uPrime_v0.8.0 ^
    --add-data "assets;assets" ^
    --add-data "version.txt;." ^
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
    --hidden-import openpyxl ^
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
    --exclude-module lvpyio ^
    --exclude-module lvpyio_wrapped ^
    main.py
echo.
if errorlevel 1 (
    echo BUILD FAILED. Check output above.
) else (
    echo BUILD COMPLETE: dist\uPrime_v0.8.0.exe
)
pause
