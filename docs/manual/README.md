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
| `\begin{notebox}…\end{notebox}` | `\begin{quote} \textbf{Note} … \end{quote}` |
| `\begin{tipbox}…\end{tipbox}` | `\begin{quote} \textbf{Tip} … \end{quote}` |
| `\begin{warnbox}…\end{warnbox}` | `\begin{quote} \textbf{Warning} … \end{quote}` |
| `\begin{stepbox}…\end{stepbox}` | `\begin{quote} \textbf{Important} … \end{quote}` |
| `\begin{mdframed}[…]…\end{mdframed}` | `\begin{quote} \textbf{Note} … \end{quote}` |
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
