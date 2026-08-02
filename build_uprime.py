#!/usr/bin/env python
"""
build_uprime.py
---------------
Single PyInstaller build script with TWO recipes from ONE source:

  PUBLIC build (default):
      python build_uprime.py
    -> uPrime_v<version>.exe
    -> lvpyio and lvpyio_wrapped are EXCLUDED (--exclude-module) so the
       restricted, non-GPL LaVision library is never swept into a public build.

  LAB / DaVis-enabled build:
      python build_uprime.py --with-lvpyio          (or UPRIME_BUNDLE_LVPYIO=1)
    -> uPrime_v<version>_davis.exe
    -> lvpyio (and lvpyio_wrapped if present) are bundled via --collect-all /
       --collect-binaries. Fails early if lvpyio is not importable here.

Distribution "Option A": a frozen exe only sees BUNDLED packages, so the public
exe can never use a separately pip-installed lvpyio. DaVis support therefore
comes only from (1) running from source with lvpyio, or (2) this --with-lvpyio
build.
"""

import argparse
import importlib.util
import os
import subprocess
import sys


def _read_version():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(here, "version.txt"), encoding="utf-8") as f:
            return f.read().strip() or "0.7.1"
    except OSError:
        return "0.7.1"


def _module_available(name):
    return importlib.util.find_spec(name) is not None


# PyInstaller args shared by both recipes (ported from build_exe.bat).
_BASE_ARGS = [
    "--onefile", "--windowed",
    "--add-data", "assets%sassets" % os.pathsep,
    "--collect-all", "matplotlib",
    "--hidden-import", "matplotlib.backends.backend_qtagg",
    "--hidden-import", "matplotlib.backends.backend_qt",
    "--hidden-import", "matplotlib.figure",
    "--collect-all", "PyQt6",
    "--hidden-import", "PyQt6.QtWidgets",
    "--hidden-import", "PyQt6.QtCore",
    "--hidden-import", "PyQt6.QtGui",
    "--hidden-import", "PyQt6.sip",
    "--collect-all", "numpy",
    "--hidden-import", "numpy.ma",
    "--collect-all", "scipy",
    "--hidden-import", "scipy.signal",
    "--hidden-import", "scipy.ndimage",
    "--hidden-import", "scipy.interpolate",
    "--hidden-import", "scipy.io",
    "--hidden-import", "scipy.io.matlab",
    "--collect-all", "pyfftw",
    "--collect-binaries", "pyfftw",
    "--collect-all", "h5py",
    "--collect-all", "PIL",
    "--hidden-import", "concurrent.futures",
    "--hidden-import", "traceback",
]


def build(with_lvpyio):
    version = _read_version()

    args = ["pyinstaller"] + list(_BASE_ARGS)

    if with_lvpyio:
        if not _module_available("lvpyio") and not _module_available("lvpyio_wrapped"):
            sys.exit(
                "ERROR: --with-lvpyio was requested but neither 'lvpyio' nor "
                "'lvpyio_wrapped' is installed in this build environment; "
                "cannot build the DaVis-enabled variant.\n"
                "Install it first:  pip install -r requirements-lvpyio.txt")
        name = f"uPrime_v{version}_davis"
        for mod in ("lvpyio", "lvpyio_wrapped"):
            if _module_available(mod):
                args += ["--collect-all", mod, "--collect-binaries", mod,
                         "--hidden-import", mod]
        variant = "LAB / DaVis-enabled (lvpyio bundled)"
    else:
        name = f"uPrime_v{version}"
        # Never let the restricted, non-GPL lvpyio into a public build.
        args += ["--exclude-module", "lvpyio",
                 "--exclude-module", "lvpyio_wrapped"]
        variant = "PUBLIC (no lvpyio)"

    args += ["--name", name, "main.py"]

    print("=" * 64)
    print(f"uPrime build — variant: {variant}")
    print(f"Output exe   : dist/{name}.exe")
    print("=" * 64)

    # Make sure PyInstaller itself is available.
    subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller",
                    "--quiet"], check=False)

    result = subprocess.run(args)
    print()
    if result.returncode != 0:
        print("BUILD FAILED. Check output above.")
        sys.exit(result.returncode)
    print(f"BUILD COMPLETE ({variant}): dist/{name}.exe")


def main():
    parser = argparse.ArgumentParser(description="Build the uPrime executable.")
    parser.add_argument(
        "--with-lvpyio", action="store_true",
        help="Bundle lvpyio for a DaVis-enabled (lab) build. Without this flag "
             "a public build is produced with lvpyio excluded.")
    ns = parser.parse_args()

    env_flag = os.environ.get("UPRIME_BUNDLE_LVPYIO", "") == "1"
    build(with_lvpyio=ns.with_lvpyio or env_flag)


if __name__ == "__main__":
    main()
