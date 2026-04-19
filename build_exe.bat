@echo off
REM ============================================================
REM  build_exe.bat -- uPrime v0.4.0
REM  Builds a single-file windowed executable via PyInstaller
REM  Target: Python 3.11
REM  Usage:  double-click or run from project root
REM
REM  Packages bundled:
REM    matplotlib  -- collect-all (backends, fonts, style data)
REM    PyQt6       -- collect-all (platform plugins, sip binding)
REM    numpy       -- collect-all (compiled extensions)
REM    scipy       -- collect-all (signal, ndimage, interpolate)
REM    pyfftw      -- collect-all + collect-binaries (FFTW DLLs)
REM    PIL/Pillow  -- collect-all (image handling)
REM
REM  New in v0.4.0:
REM    assets/     -- logo + manual PDF bundled via --add-data
REM    scipy.linalg, scipy.sparse -- DMD SVD/eigendecomposition
REM    PyQt6.QtSvg -- SVG logo support
REM ============================================================
python -m pip install pyinstaller --quiet
pyinstaller ^
    --onefile ^
    --windowed ^
    --name uPrime_v0.4.0 ^
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
    --collect-all pyfftw ^
    --collect-binaries pyfftw ^
    --hidden-import pyfftw ^
    --hidden-import pyfftw.interfaces ^
    --hidden-import pyfftw.interfaces.numpy_fft ^
    --hidden-import pyfftw.interfaces.scipy_fft ^
    --hidden-import pyfftw.interfaces.cache ^
    --collect-all PIL ^
    --hidden-import PIL.Image ^
    --hidden-import PIL.ImageDraw ^
    --hidden-import concurrent.futures ^
    --hidden-import csv ^
    --hidden-import datetime ^
    --hidden-import re ^
    --hidden-import traceback ^
    main.py
echo.
if errorlevel 1 (
    echo BUILD FAILED. Check output above.
) else (
    echo BUILD COMPLETE: dist\uPrime_v0.4.0.exe
)
pause