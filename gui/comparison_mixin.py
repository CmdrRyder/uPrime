"""
gui/comparison_mixin.py
-----------------------
Mixin that adds multi-case comparison capability to any analysis window.

Usage
-----
class MyWindow(ComparisonMixin, QWidget):
    def __init__(self, ...):
        super().__init__(...)
        ...
        self._init_comparison_toolbar(my_layout)

    def _validate_csv(self, df) -> bool:   # required
        ...

    def _plot_comparison(self, quantities, layout_mode):  # required
        ...
"""

import csv
import os
import numpy as np
from PyQt6.QtWidgets import (
    QHBoxLayout, QPushButton, QFileDialog, QMessageBox,
    QDialog, QVBoxLayout, QFormLayout, QLineEdit,
)

_TAB10 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# ---------------------------------------------------------------------------
# Minimal pandas-compatible column/table types (no pandas required)
# ---------------------------------------------------------------------------

class _Series:
    """
    Single named column — drop-in for pandas.Series for the operations used
    here: .values (numpy array), iteration, indexing, and len().
    """
    __slots__ = ("_arr",)

    def __init__(self, data):
        try:
            self._arr = np.asarray(data, dtype=np.float64)
        except (ValueError, TypeError):
            # Retry element-wise: empty / None → NaN so numeric columns stay
            # float64.  If a genuine non-numeric string is found, fall back to
            # object (e.g. a "case" name column).
            converted = []
            non_numeric = False
            for v in data:
                if v is None or (isinstance(v, str) and v.strip() == ""):
                    converted.append(np.nan)
                else:
                    try:
                        converted.append(float(v))
                    except (ValueError, TypeError):
                        non_numeric = True
                        break
            if non_numeric:
                self._arr = np.asarray(data, dtype=object)
            else:
                self._arr = np.asarray(converted, dtype=np.float64)

    @property
    def values(self):
        return self._arr

    def __iter__(self):
        return iter(self._arr)

    def __len__(self):
        return len(self._arr)

    def __getitem__(self, key):
        return self._arr[key]

    def __array__(self, dtype=None):
        return self._arr if dtype is None else self._arr.astype(dtype)


class _SimpleDF:
    """
    Minimal column-oriented table — drop-in for pandas.DataFrame for the
    operations used here: .columns, ["col"], .insert(), and len().
    """

    def __init__(self, columns, rows):
        self._col_order = list(columns)
        self._data = {}
        n_cols = len(columns)
        for i, col in enumerate(columns):
            vals = [row[i] if i < len(row) else "" for row in rows]
            self._data[col] = _Series(vals)

    @property
    def columns(self):
        return self._col_order

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        if not self._data:
            return 0
        return len(next(iter(self._data.values())))

    def insert(self, pos, name, value):
        """Insert a new column at *pos* filled with *value* (scalar or array)."""
        n = len(self)
        self._col_order.insert(pos, name)
        if np.ndim(value) == 0:
            data = [value] * n
        else:
            data = list(value)
        self._data[name] = _Series(data)


def _read_csv(path):
    """
    Read a CSV file into a _SimpleDF, skipping blank lines and lines whose
    first field starts with '#'.  Replaces pd.read_csv(path, comment='#').
    """
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header = None
        rows = []
        for raw in reader:
            if not raw:
                continue
            if raw[0].strip().startswith("#"):
                continue
            stripped = [c.strip() for c in raw]
            if header is None:
                header = stripped
            else:
                rows.append(stripped)
    if header is None:
        raise ValueError("No data found in CSV file")
    return _SimpleDF(header, rows)


def _concat_to_csv(frames, path):
    """
    Write multiple _SimpleDF objects (same column layout) to a single CSV.
    Replaces pd.concat(frames, ignore_index=True).to_csv(path, index=False).
    """
    if not frames:
        return
    header = list(frames[0].columns)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for df in frames:
            col_arrs = [df[col].values for col in header]
            for row in zip(*col_arrs):
                writer.writerow(row)


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------

class ComparisonMixin:
    """
    Mixin providing import, compare, style-edit, and bulk-export for
    multi-case line-profile comparisons.  Subclasses must implement
    _validate_csv(df) and _plot_comparison(quantities, layout_mode).
    """

    _axis_columns = []   # subclasses override: columns treated as x-axis, excluded from selector

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #

    def _init_comparison_toolbar(self, parent_layout):
        self._ComparisonMixin__current_case_name = "Current"
        self._comparison_case_mgr_dlg = None
        self._cases = []
        self._last_plot_args = None

        row = QHBoxLayout()
        self._btn_import        = QPushButton("Import Case")
        self._btn_compare       = QPushButton("Compare")
        self._btn_edit_styles   = QPushButton("Edit Styles")
        self._btn_manage_cases  = QPushButton("Manage Cases")
        self._btn_export_all    = QPushButton("Export All Cases")

        self._btn_import.clicked.connect(self._on_import_case)
        self._btn_compare.clicked.connect(self._on_compare)
        self._btn_edit_styles.clicked.connect(self._open_case_manager)
        self._btn_manage_cases.clicked.connect(self._open_case_manager)
        self._btn_export_all.clicked.connect(self._on_export_all)

        row.addWidget(self._btn_import)
        row.addWidget(self._btn_compare)
        row.addWidget(self._btn_edit_styles)
        row.addWidget(self._btn_manage_cases)
        row.addWidget(self._btn_export_all)
        row.addStretch()
        parent_layout.addLayout(row)

        self._update_comparison_toolbar()

    def _update_comparison_toolbar(self):
        enough = len(self._cases) >= 2
        self._btn_edit_styles.setEnabled(enough)
        self._btn_manage_cases.setEnabled(bool(self._cases))
        self._btn_export_all.setEnabled(enough)

    def _next_color(self):
        used = {c["color"] for c in self._cases}
        for color in _TAB10:
            if color not in used:
                return color
        return _TAB10[len(self._cases) % 10]

    # ------------------------------------------------------------------ #
    # _current_case_name property
    # ------------------------------------------------------------------ #

    @property
    def _current_case_name(self):
        return self._ComparisonMixin__current_case_name

    @_current_case_name.setter
    def _current_case_name(self, value):
        self._ComparisonMixin__current_case_name = value

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _get_session_case_name(self):
        """Return any already-set session case name, checking several sources."""
        try:
            val = self._session_case_name
            if val:
                return val
        except AttributeError:
            pass

        try:
            w = self.parent()
            while w is not None:
                if hasattr(w, "session_case_name") and w.session_case_name:
                    return w.session_case_name
                w = w.parent()
        except Exception:
            pass

        try:
            from PyQt6.QtWidgets import QApplication
            aw = QApplication.instance().activeWindow()
            if aw is not None and aw._session_case_name:
                return aw._session_case_name
        except Exception:
            pass

        return "Current"

    def _show_import_summary(self, n_ok, errors):
        lines = [f"Successfully imported: {n_ok} file(s)."]
        if errors:
            lines.append("")
            lines.append("Failures:")
            for fname, msg in errors:
                lines.append(f"  • {fname}: {msg}")
            QMessageBox.warning(self, "Import Summary", "\n".join(lines))
        else:
            QMessageBox.information(self, "Import Summary", "\n".join(lines))

    # ------------------------------------------------------------------ #
    # Import (multi-file)
    # ------------------------------------------------------------------ #

    def _on_import_case(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import Case CSV(s)", "", "CSV Files (*.csv)")
        if not paths:
            return

        is_first_import = (
            len(self._cases) == 0 and self._current_case_name == "Current"
        )

        newly_added = []
        errors = []

        for path in paths:
            basename = os.path.basename(path)
            df = None
            try:
                df = _read_csv(path)
            except Exception as exc:
                errors.append((basename, str(exc)))

            if df is None:
                continue

            if not self._check_module_columns(df):
                continue

            auto_name = f"Data {len(self._cases) + len(newly_added) + 1}"
            newly_added.append(({
                "name":      auto_name,
                "data":      df,
                "color":     None,
                "linestyle": "-",
                "source":    path,
            }, basename))

        if not newly_added:
            self._show_import_summary(0, errors)
            return

        # --- Naming dialog ---
        dlg = QDialog(self)
        dlg.setWindowTitle("Name Imported Cases")
        dlg.setMinimumWidth(440)
        vlay = QVBoxLayout(dlg)
        vlay.setContentsMargins(12, 12, 12, 12)
        vlay.setSpacing(8)
        form = QFormLayout()
        form.setSpacing(6)

        current_edit = None
        if is_first_import:
            current_default = self._get_session_case_name()
            current_edit = QLineEdit(current_default)
            form.addRow("Current dataset name:", current_edit)

        name_edits = []
        for case, basename in newly_added:
            edit = QLineEdit(case["name"])
            form.addRow(basename + ":", edit)
            name_edits.append(edit)

        vlay.addLayout(form)
        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(dlg.accept)
        vlay.addWidget(btn_ok)
        dlg.exec()

        if is_first_import and current_edit is not None:
            val = current_edit.text().strip()
            self._current_case_name = val if val else self._get_session_case_name()

        for (case, _), edit in zip(newly_added, name_edits):
            val = edit.text().strip()
            if val:
                case["name"] = val

        for case, _ in newly_added:
            case["color"] = self._next_color()
            self._cases.append(case)

        self._update_comparison_toolbar()
        self._show_import_summary(len(newly_added), errors)

    def _check_module_columns(self, df):
        """
        Calls _validate_csv(df).  If it returns False and the file is missing
        the columns declared in _expected_columns, shows a specific warning
        dialog so the user knows why the file was rejected.  If the expected
        columns *are* present but validation still fails (e.g. SpectraWindow
        tab-3 "not supported"), the subclass has already shown its own message
        and this method stays silent.
        """
        if self._validate_csv(df):
            return True
        expected = getattr(self, "_expected_columns", [])
        if expected and not set(expected).issubset(set(df.columns)):
            module_name = getattr(self, "_module_name", type(self).__name__)
            QMessageBox.warning(
                self, "Wrong File Type",
                f"This file does not appear to be a {module_name} export.\n\n"
                f"Expected columns: {expected}\n"
                f"Found columns:    {list(df.columns)}\n\n"
                "Skipping this file."
            )
        return False

    def _validate_csv(self, df):
        raise NotImplementedError(
            f"{type(self).__name__} must implement _validate_csv(df) -> bool")

    # ------------------------------------------------------------------ #
    # Compare
    # ------------------------------------------------------------------ #

    def _on_compare(self):
        import traceback
        try:
            if len(self._cases) < 2:
                QMessageBox.information(
                    self, "Not Enough Cases",
                    "Import at least one case before comparing.")
                return
            axis_cols = set(getattr(self, "_axis_columns", []))
            col_sets = [set(c["data"].columns) - axis_cols for c in self._cases]
            quantity_names = sorted(set.intersection(*col_sets)) if col_sets else []
            if not quantity_names:
                QMessageBox.warning(self, "No Common Quantities",
                    "No common data columns found across the imported cases.")
                return
            module_name = getattr(self, "_module_name", type(self).__name__)
            from gui.quantity_selector_dialog import QuantitySelectorDialog
            dlg = QuantitySelectorDialog(quantity_names, module_name, parent=self)
            if not dlg.exec():
                return
            quantities, layout_mode = dlg.get_selection()
            if not quantities:
                return
            self._last_plot_args = (quantities, layout_mode)
            self._plot_comparison(quantities, layout_mode)
        except Exception:
            QMessageBox.critical(self, "Compare Error", traceback.format_exc())

    def _plot_comparison(self, selected_quantities, layout_mode):
        raise NotImplementedError(
            f"{type(self).__name__} must implement "
            "_plot_comparison(selected_quantities, layout_mode)")

    # ------------------------------------------------------------------ #
    # Case manager (shared by Edit Styles and Manage Cases buttons)
    # ------------------------------------------------------------------ #

    def _open_case_manager(self):
        from gui.case_manager_dialog import CaseManagerDialog
        if self._comparison_case_mgr_dlg is None:
            self._comparison_case_mgr_dlg = CaseManagerDialog(
                parent=self, cases=self._cases)
            self._comparison_case_mgr_dlg.cases_changed.connect(
                self._on_case_mgr_cases_changed)
        else:
            self._comparison_case_mgr_dlg.refresh()
        self._comparison_case_mgr_dlg.show()
        self._comparison_case_mgr_dlg.raise_()
        self._comparison_case_mgr_dlg.activateWindow()

    def _on_case_mgr_cases_changed(self):
        self._update_comparison_toolbar()
        if self._last_plot_args is not None and len(self._cases) >= 2:
            try:
                self._plot_comparison(*self._last_plot_args)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Export all
    # ------------------------------------------------------------------ #

    def _on_export_all(self):
        if not self._cases:
            QMessageBox.information(self, "No Cases", "No cases to export.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export All Cases", "comparison_all_cases.csv",
            "CSV Files (*.csv)")
        if not path:
            return

        frames = []
        missing = []
        for case in self._cases:
            src = case["source"]
            if not os.path.isfile(src):
                missing.append(case["name"])
                continue
            try:
                df = _read_csv(src)
                df.insert(0, "case", case["name"])
                frames.append(df)
            except Exception as exc:
                QMessageBox.warning(self, "Read Error",
                    f"Could not read source for '{case['name']}':\n{exc}")

        if missing:
            QMessageBox.warning(self, "Missing Source Files",
                "Could not locate source files for:\n"
                + "\n".join(f"  • {n}" for n in missing))

        if not frames:
            QMessageBox.critical(self, "Export Failed",
                "No data could be read — nothing written.")
            return

        try:
            _concat_to_csv(frames, path)
            QMessageBox.information(self, "Export Complete",
                f"Exported {len(frames)} case(s) to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", str(exc))
