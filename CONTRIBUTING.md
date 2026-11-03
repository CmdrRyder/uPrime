# Contributing to uPrime

Thanks for your interest in uPrime. This document covers the development
setup, the repository layout, how to run the tests, how to build the
executables, and the checklist for a version bump.

uPrime is GPL-3.0-only. By contributing you agree that your contribution is
licensed under the same terms.

---

## Development setup

```bash
git clone https://github.com/CmdrRyder/uPrime.git
cd uPrime
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

Then launch the app either way:

```bash
uprime              # console script installed by pip
python main.py      # thin launcher, equivalent
```

Optional extras:

| Extra | Install | What it adds |
|---|---|---|
| `fftw` | `pip install -e ".[fftw]"` | pyfftw-backed FFT for the 3-D spatial spectra module. Imported behind `try/except`, so the app runs without it. |
| `dev` | `pip install -e ".[dev]"` | pytest. |
| DaVis | `pip install -r requirements-lvpyio.txt` | LaVision `lvpyio` for `.set`/`.vc7`/`.vec` input. Separately licensed, **not** GPL — see that file's header. |

`requirements.txt` is kept alongside `pyproject.toml` on purpose: the public
PyInstaller build installs from it so that pyfftw is always bundled in the
released executable, even though it is an optional extra for a source install.

---

## Repository layout

```
uprime/              the installable package
  app.py             matplotlib defaults + main(); the real entry point
  core/              computation, I/O, no Qt imports
  gui/               PyQt6 windows, one module per analysis window
  assets/            logos + bundled uPrime_Manual.pdf
  version.txt        single source of truth for the version
main.py              thin launcher (python main.py, and PyInstaller's target)
tests/               pytest suite
build_uprime.py      cross-platform PyInstaller driver
build_exe.bat        Windows public build (no lvpyio)
```

Two conventions worth knowing:

- **`core/` never imports from `gui/`.** Computation stays headless and
  testable; only `gui/` may import `core/`.
- **Asset and version lookup is layout-sensitive.** `_asset_path()` and
  `_read_app_version()` in `uprime/gui/main_window.py` resolve to the parent of
  `gui/` (i.e. `uprime/`) from source, and to `sys._MEIPASS` when frozen. That
  is why the build recipes bundle `uprime/assets` to `assets` and
  `uprime/version.txt` to `.` — the source path changed when the package moved,
  the destinations inside the bundle must not.

---

## Running the tests

```bash
pytest                      # whole suite
pytest -k "not GUI"         # core modules only
pytest -k "GUI"             # GUI smoke tests only
```

Tests run headlessly (matplotlib Agg + offscreen Qt) and need no real `.dat`
files. GUI tests are skipped automatically where no Qt platform is available.

---

## Building the executables

Two recipes, and the distinction matters legally:

| Build | Command | lvpyio / DaVis | Redistributable |
|---|---|---|---|
| Public | `build_exe.bat` or `python build_uprime.py` | no | yes — this is the release build |
| TFML-internal | `python build_uprime.py --with-lvpyio` | yes (bundled) | **no** |

The internal build bundles LaVision's proprietary `lvpyio`. It must not be
attached to a GitHub release or shared outside TFML.

---

## Version bump checklist

The runtime version comes from `uprime/version.txt`, which `pyproject.toml`
also reads as the package version. Bumping a release means editing these:

- [ ] `uprime/version.txt` — the source of truth; everything at runtime reads this
- [ ] `CITATION.cff` — `version:` and `date-released:` (and the DOI for a Zenodo release)
- [ ] `CHANGELOG.md` — new section for the release
- [ ] `build_exe.bat` — `--name`, the header comment, and the final `BUILD COMPLETE` echo
- [ ] `.github/workflows/build-test.yml` — the `--name` in the PyInstaller step
- [ ] `README.md` — version badge, Development Status, citation line
- [ ] `index.html` — version badge and the download button text
- [ ] `uprime/gui/main_window.py` — the `_read_app_version(default=...)` fallback
- [ ] `build_uprime.py` — the `_read_version()` fallback

Do not commit a version bump and a feature change together; keep the bump its
own commit.

---

## Pull requests

- Branch off `main`.
- Keep `core/` free of Qt imports.
- Add or update tests for behavioral changes, and make sure `pytest` is green.
- Note user-visible changes in `CHANGELOG.md` under an `## [Unreleased]` heading.
