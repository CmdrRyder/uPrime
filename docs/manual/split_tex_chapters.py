#!/usr/bin/env python3
"""
Split main_0.6.0.tex into individual chapter files.
Usage: python split_tex_chapters.py [input_file] [output_dir]
"""

import re
import os
import sys

def clean_filename(title):
    clean = re.sub(r'[^a-zA-Z0-9\s\-&]', '', title)
    clean = clean.replace(' ', '_').replace('&', 'and').replace('--', '-')
    clean = clean.strip('_')
    return clean.lower()

def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else '../../uPrime_Documentation/_main.tex'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'user_guide'

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    doc_start = content.find('\\begin{document}')
    doc_end = content.find('\\end{document}')

    if doc_start == -1 or doc_end == -1:
        print("Error: Missing \\begin{document} or \\end{document}")
        sys.exit(1)

    preamble = content[:doc_start + len('\\begin{document}')]
    doc_body = content[doc_start + len('\\begin{document}'):doc_end]

    chapter_matches = list(re.finditer(r'\\chapter(\*?)\{([^}]*)\}', doc_body))
    if not chapter_matches:
        print("Error: No chapters found")
        sys.exit(1)

    appendix_idx = doc_body.find('\\appendix')
    if appendix_idx == -1:
        appendix_idx = len(doc_body)

    main_chapter_count = sum(1 for m in chapter_matches if m.start() < appendix_idx)

    os.makedirs(output_dir, exist_ok=True)
    includes = []

    # --- Special case: extract the Abstract section ---
    abstract_match = re.search(r'\\section\*?\{Abstract\}', doc_body)
    if abstract_match:
        abs_start = abstract_match.start()
        # Abstract ends at the next \chapter, \tableofcontents, or end of body
        next_chapter = chapter_matches[0].start() if chapter_matches else len(doc_body)
        toc_idx = doc_body.find('\\tableofcontents', abs_start)
        abs_end = min(next_chapter, toc_idx if toc_idx != -1 else len(doc_body))
        abstract_text = doc_body[abs_start:abs_end].strip() + '\n'
        abstract_file = os.path.join(output_dir, '00_abstract.tex')
        with open(abstract_file, 'w', encoding='utf-8') as f:
            f.write(abstract_text)
        includes.append(f"\\include{{{output_dir}/00_abstract}}")
        print(f"Created: 00_abstract.tex")

    for i, match in enumerate(chapter_matches):
        start = match.start()
        end = chapter_matches[i + 1].start() if (i + 1) < len(chapter_matches) else len(doc_body)
        chapter_text = doc_body[start:end]
        title = match.group(2)

        clean_title = clean_filename(title)
        is_appendix = start >= appendix_idx
        if is_appendix:
            local_idx = i - main_chapter_count
            prefix = f"appendix_{chr(65 + local_idx)}"
        else:
            prefix = f"{i+1:02d}"

        filename = f"{prefix}_{clean_title}.tex"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(chapter_text)
        includes.append(f"\\include{{{output_dir}/{os.path.splitext(filename)[0]}}}")
        print(f"Created: {filename}")

    # Front matter: everything before the abstract (or before first chapter if no abstract)
    fm_end = chapter_matches[0].start()
    if abstract_match and abstract_match.start() < fm_end:
        fm_end = abstract_match.start()
    front_matter = doc_body[:fm_end]
    new_main = preamble + '\n\n' + front_matter + '\n\n' + '\n'.join(includes) + '\n\n\\end{document}\n'

    with open(os.path.join(os.getcwd(), 'main.tex'), 'w', encoding='utf-8') as f:
        f.write(new_main)
    print(f"New main file: {os.getcwd()}/main.tex")

    print("\n" + "="*70)
    print("INCLUDE COMMANDS:")
    print("="*70)
    for inc in includes:
        print(inc)

if __name__ == '__main__':
    main()
