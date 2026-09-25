"""Compile rapport/rapport.typ en rapport/rapport.pdf.

Polices intégrées à Typst uniquement : le PDF est identique sur toutes les machines.
"""

from pathlib import Path

import typst

HERE = Path(__file__).resolve().parent

typst.compile(str(HERE / "rapport.typ"), output=str(HERE / "rapport.pdf"),
              root=str(HERE.parent), ignore_system_fonts=True)
print(f"écrit {HERE / 'rapport.pdf'}")
