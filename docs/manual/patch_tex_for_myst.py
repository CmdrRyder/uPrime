#!/usr/bin/env python3
"""Patch .tex files in-place to be MyST-compatible.

Converts custom LaTeX macros/environments that MyST can't parse into
equivalent MyST-native constructs so content is not silently dropped.
"""

import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def match_braces(text: str, start: int) -> int:
    """Given index of opening '{', return index of matching '}'."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"Unmatched braces starting at index {start}")


def replace_macro(text: str, macro: str, replacement_fn) -> str:
    """Replace all \\macro{...} occurrences, handling nested braces."""
    pattern = '\\' + macro + '{'
    result = []
    i = 0
    while i < len(text):
        idx = text.find(pattern, i)
        if idx == -1:
            result.append(text[i:])
            break
        result.append(text[i:idx])
        brace_start = idx + len(pattern) - 1  # index of '{'
        brace_end = match_braces(text, brace_start)
        inner = text[brace_start + 1:brace_end]
        result.append(replacement_fn(inner))
        i = brace_end + 1
    return ''.join(result)


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------

# Labels must be bare keywords (no colon) matching MyST's GitHub-style
# admonition kinds: note, tip, important, warning, caution.
# myst-transforms converts a blockquote starting with \textbf{<keyword>}
# into a real admonition node.
ADMONITION_MAP = {
    'notebox': 'Note',
    'tipbox': 'Tip',
    'warnbox': 'Warning',
    'stepbox': 'Important',
    'mdframed': 'Note',
}


def convert_admonitions(text: str) -> str:
    """Convert \\begin{xbox}...\\end{xbox} and mdframed to quote + textbf admonitions.

    MyST's LaTeX parser (tex-to-myst) does not support `framed` or backtick
    admonition fences inside .tex files. The `quote` environment renders as a
    blockquote, and myst-transforms' GitHub-bold-admonition rule converts a
    blockquote whose first paragraph starts with \\textbf{Note} (etc.) into a
    real admonition node. The label must be a bare keyword without a colon.
    """
    for env, label in ADMONITION_MAP.items():
        # mdframed may have optional [args]
        if env == 'mdframed':
            begin_pat = re.compile(
                r'\\begin\{mdframed\}\[.*?\]\n?', re.DOTALL
            )
        else:
            begin_pat = re.compile(r'\\begin\{' + env + r'\}')
        end_pat = re.compile(r'\\end\{' + env + r'\}')

        while True:
            bmatch = begin_pat.search(text)
            if not bmatch:
                break
            ematch = end_pat.search(text, bmatch.end())
            if not ematch:
                break
            inner = text[bmatch.end():ematch.start()]
            # Strip leading/trailing blank lines from inner content
            inner = inner.strip('\n')
            replacement = f'\\begin{{quote}}\n\\textbf{{{label}}}\n{inner}\n\\end{{quote}}'
            text = text[:bmatch.start()] + replacement + text[ematch.end():]
    return text


def convert_uipth(text: str) -> str:
    """\\uipth{...} → `...` (inline code for UI elements)."""
    return replace_macro(text, 'uipth', lambda inner: f'`{inner}`')


def convert_software(text: str) -> str:
    """\\software{uPrime} → \\href{url}{uPrime} (hyperlink with anchor "uPrime")."""
    return replace_macro(text, 'software', lambda inner: f'\\href{{https://cmdrryder.github.io/uPrime/}}{{{inner}}}' if inner.strip() == 'uPrime' else inner)


def convert_key(text: str) -> str:
    """\\key{...} → `...` (inline code for keyboard keys)."""
    return replace_macro(text, 'key', lambda inner: f'`{inner}`')


def convert_textsf(text: str) -> str:
    """\\textsf{...} → plain text."""
    return replace_macro(text, 'textsf', lambda inner: inner)


def remove_addlinespace(text: str) -> str:
    """Remove \\addlinespace lines entirely."""
    return re.sub(r'^\s*\\addlinespace\s*\n', '', text, flags=re.MULTILINE)


def convert_fcolorbox(text: str) -> str:
    """\\fcolorbox{...}{...}{content} → just content."""
    while True:
        idx = text.find('\\fcolorbox{')
        if idx == -1:
            break
        # Parse three brace groups
        pos = idx + len('\\fcolorbox')
        groups = []
        for _ in range(3):
            if pos >= len(text) or text[pos] != '{':
                break
            end = match_braces(text, pos)
            groups.append(text[pos + 1:end])
            pos = end + 1
        if len(groups) < 3:
            break
        content = groups[2]
        text = text[:idx] + content + text[pos:]
    return text


def convert_parbox(text: str) -> str:
    """\\parbox{...}{content} → just content."""
    while True:
        idx = text.find('\\parbox{')
        if idx == -1:
            break
        pos = idx + len('\\parbox')
        groups = []
        for _ in range(2):
            if pos >= len(text) or text[pos] != '{':
                break
            end = match_braces(text, pos)
            groups.append(text[pos + 1:end])
            pos = end + 1
        if len(groups) < 2:
            break
        content = groups[1]
        text = text[:idx] + content + text[pos:]
    return text


ACCENT_MAP = {
    "\\'{A}": 'Á', "\\'{a}": 'á',
    "\\'{E}": 'É', "\\'{e}": 'é',
    "\\'{I}": 'Í', "\\'{i}": 'í',
    "\\'{O}": 'Ó', "\\'{o}": 'ó',
    "\\'{U}": 'Ú', "\\'{u}": 'ú',
    "\\'{C}": 'Ç', "\\'{c}": 'ç',
    "\\'{N}": 'Ñ', "\\'{n}": 'ñ',
}


def convert_accents(text: str) -> str:
    """Replace LaTeX accent commands with Unicode equivalents."""
    for latex, unicode in ACCENT_MAP.items():
        text = text.replace(latex, unicode)
    return text


def fix_color_mixing(text: str) -> str:
    """Replace xcolor mixing like 'upblue!80!black' with a plain color name."""
    text = re.sub(r'upblue!\d+!black', 'upblue', text)
    text = re.sub(r'upblue!\d+', 'upblue', text)
    return text


def demote_headings(text: str) -> str:
    """Demote heading levels by one: subsection→subsubsection, section→subsection, chapter→section.

    Must be applied deepest-first to avoid cascading (e.g. section→subsection
    then that new subsection→subsubsection).
    """
    # \subsection (and \subsection*) → \subsubsection (deepest first)
    text = re.sub(r'\\subsection\*', r'\\subsubsection*', text)
    text = re.sub(r'\\subsection\b', r'\\subsubsection', text)
    # \section (and \section*) → \subsection
    text = re.sub(r'\\section\*', r'\\subsection*', text)
    text = re.sub(r'\\section\b', r'\\subsection', text)
    # \chapter → \section
    text = re.sub(r'\\chapter\b', r'\\section', text)
    return text


def convert_fa_icons(text: str) -> str:
    r"""\faExclamationTriangle (fontawesome) -> a warning sign, eating a following \quad.

    MyST supports neither macro in text mode. The pattern is matched as a pair,
    not \quad on its own, so \quad and \qquad inside maths are untouched.
    """
    return re.sub(r'\\faExclamationTriangle(\\quad(?![a-zA-Z])\s*)?', '⚠ ', text)


def remove_includes(text: str) -> str:
    """Remove \\include{...} lines (used in main.tex; each file is built independently by MyST)."""
    return re.sub(r'^\s*\\include\{[^}]*\}\s*\n', '', text, flags=re.MULTILINE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TRANSFORMS = [
    convert_admonitions,
    convert_uipth,
    convert_software,
    convert_key,
    convert_textsf,
    convert_fcolorbox,
    convert_parbox,
    remove_addlinespace,
    convert_accents,
    fix_color_mixing,
    convert_fa_icons,
    demote_headings,
]


def patch_file(path: Path) -> dict:
    """Apply all transforms to a single file. Returns stats."""
    original = path.read_text(encoding='utf-8')
    content = original
    stats = {}
    for fn in TRANSFORMS:
        new_content = fn(content)
        if new_content != content:
            # Count how many replacements happened
            stats[fn.__name__] = len(re.findall(r'```|\\uipth|\\software|\\key|\\textsf|\\fcolorbox|\\addlinespace|\\begin\{(notebox|tipbox|warnbox|stepbox|mdframed)\}|\\\'|upblue!', content))
            content = new_content
    if content != original:
        path.write_text(content, encoding='utf-8')
    return stats


def main():
    manual_dir = Path(__file__).parent
    tex_files = sorted(manual_dir.glob('user_guide/*.tex')) + [manual_dir / 'main.tex']

    # Dry-run mode
    dry_run = '--dry-run' in sys.argv

    total_stats = {}
    for f in tex_files:
        if not f.exists():
            continue
        if dry_run:
            original = f.read_text(encoding='utf-8')
            content = original
            changes = []
            for fn in TRANSFORMS:
                new_content = fn(content)
                if new_content != content:
                    changes.append(fn.__name__)
                    content = new_content
            if changes:
                print(f'  Would patch: {f.relative_to(manual_dir)}  [{', '.join(changes)}]')
                total_stats[str(f.relative_to(manual_dir))] = changes
        else:
            stats = patch_file(f)
            if stats:
                print(f'  Patched: {f.relative_to(manual_dir)}  [{', '.join(stats.keys())}]')
                total_stats[str(f.relative_to(manual_dir))] = list(stats.keys())

    if total_stats:
        print(f'\n{"Would patch" if dry_run else "Patched"} {len(total_stats)} file(s).')
    else:
        print('\nNo changes needed.')


if __name__ == '__main__':
    main()
