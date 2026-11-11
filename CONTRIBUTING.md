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
uprime-piv          # alias of the same command (since v0.9.2)
python main.py      # thin launcher, equivalent
```

The package name and the command differ: users `pip install uprime-piv` from
PyPI and run `uprime`. `uprime-piv` is registered as an alias in
`pyproject.toml` so that typing the package name also works.

Optional extras:

| Extra | Install | What it adds |
|---|---|---|
| `dev` | `pip install -e ".[dev]"` | pytest. |
| `fft` | `pip install -e ".[fft]"` | pyFFTW, for the **Spatial E(k) using FFT** tab. Optional so uPrime installs on a Python with no pyFFTW wheel. |
| `all` | `pip install -e ".[all]"` | Every optional runtime extra; currently the same as `fft`. |
| DaVis | `pip install -r requirements-lvpyio.txt` | LaVision `lvpyio` for `.set`/`.vc7`/`.vec` input. Separately licensed, **not** GPL — see that file's header. |

`requirements.txt` is kept alongside `pyproject.toml` on purpose: the public
PyInstaller build installs from it, so the released executable always bundles
the same pinned set.

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
- [ ] `uprime/app.py` — the version line in the module docstring
- [ ] `uprime/gui/main_window.py` — the `_read_app_version(default=...)` fallback
- [ ] `build_uprime.py` — both `_read_version()` fallbacks
- [ ] `build_exe.bat` — the header comment, `--name`, the final `BUILD COMPLETE`
      echo, and a new "New in" REM block (keep the older ones)
- [ ] `.github/workflows/build-test.yml` — the `--name` in the PyInstaller step
- [ ] `build_exe_TFML.bat` (local, gitignored) — same edits as `build_exe.bat`
- [ ] `README.md` — Development Status and citation line. There is no
      version badge to bump; the PyPI badge updates itself
- [ ] `README.md` — the five `raw/vX.Y.Z/` image links and three
      `blob/vX.Y.Z/` file links, pinned to the new tag. The README is
      frozen into each PyPI release, so pinning keeps old descriptions
      working if files move on `main`
- [ ] `index.html` — version badge, the download button text, and a "NEW IN"
      block (keep the older ones)
- [ ] `CITATION.cff` — `version:` and `date-released:` (keep the concept DOI,
      which covers every version)
- [ ] `pyproject.toml` — the `Development Status` classifier, if the maturity
      changed
- [ ] `CHANGELOG.md` — rename `[Unreleased]` to the new version and date

Leave historical references alone: past `CHANGELOG.md` entries, the "(vX.Y)"
markers in `uprime/gui/probability_window.py` and the tests that record when a
feature arrived, the older REM and "NEW IN" blocks, the old `.spec` files, the
manual and the screenshot filenames.

Do not commit a version bump and a feature change together; keep the bump its
own commit.

---

## Pull requests

- Branch off `main`.
- Keep `core/` free of Qt imports.
- Add or update tests for behavioral changes, and make sure `pytest` is green.
- Update [`docs/API.md`](docs/API.md) whenever a public function in `uprime/core`
  changes -- its name, parameters, or what it returns.
- Note user-visible changes in `CHANGELOG.md` under an `## [Unreleased]` heading.
