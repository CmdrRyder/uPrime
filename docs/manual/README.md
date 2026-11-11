# Manual build scripts

Two Python scripts automate the workflow from a monolithic LaTeX file to a
MyST-built web manual.

## 1. Split the monolithic `.tex` into chapter files

```bash
python3 split_tex_chapters.py [input.tex] [output_dir]
```

- **Default input:** `../../uPrime_Documentation/_main.tex`
- **Default output:** `user_guide/`

Splits a single `.tex` file on `\chapter{...}` boundaries into individual
files under `user_guide/`.

## 2. Patch `.tex` files for MyST compatibility

```bash
python3 patch_tex_for_myst.py            # apply changes in-place
python3 patch_tex_for_myst.py --dry-run  # preview without writing
```

Transforms custom LaTeX constructs that MyST can't parse:

| LaTeX | Becomes |
|-------|---------|
| `\begin{notebox}…\end{notebox}` | ` ```{note} … ``` ` |
| `\begin{tipbox}…\end{tipbox}` | ` ```{tip} … ``` ` |
| `\begin{warnbox}…\end{warnbox}` | ` ```{warning} … ``` ` |
| `\begin{stepbox}…\end{stepbox}` | ` ```{note} … ``` ` |
| `\begin{mdframed}[…]…\end{mdframed}` | ` ```{note} … ``` ` |
| `\uipth{X}` | `` `X` `` |
| `\software{uPrime}` | `\href{https://cmdrryder.github.io/uPrime/}{uPrime}` |
| `\chapter{X}` | `\section{X}` (headings demoted one level) |
| `\fcolorbox{..}{..}{X}` / `\parbox{..}{X}` | `X` |
| `\'{A}`, `\'{e}`, … | Unicode `Á`, `é`, … |

## 3. Build the site

```bash
myst build
```

Output goes to `_build/site/`.

## Full pipeline (from scratch)

```bash
python3 split_tex_chapters.py
python3 patch_tex_for_myst.py
myst build
```
