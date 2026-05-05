# conftest.py (Projekt-Wurzel)
#
# Sorgt dafuer, dass pytest das Package "vigilex" findet,
# auch ohne dass vigilex per "pip install -e ." installiert ist.
# pytest laedt diese Datei automatisch, bevor Tests ausgefuehrt werden.

import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
