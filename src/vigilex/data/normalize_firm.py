import re

LEGAL_SUFFIXES = re.compile(
    r'\s+(inc|incorporated|llc|corp|corporation|co|ltd|gmbh|ag)\.?$',
    flags=re.IGNORECASE
)

def normalize_firm(name):
    """Normalise manufacturer name for fuzzy matching."""
    if not isinstance(name, str):
        return ''
    name = name.strip().upper()
    name = re.sub(r'[,\.]', '', name)
    name = LEGAL_SUFFIXES.sub('', name)
     
    return name.strip()