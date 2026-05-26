"""
patch_golden_set_acceptable.py
Adds acceptable_pt_codes to specific cases in golden_set_v1.jsonl.
Idempotent: re-running overwrites existing acceptable_pt_codes with the values below.
All other cases and fields are unchanged.

Run: python3 scripts/patch_golden_set_acceptable.py
"""
import json
import shutil
from pathlib import Path

GOLDEN_SET = Path("data/eval/golden_set_v1.jsonl")
BACKUP     = Path("data/eval/golden_set_v1.jsonl.bak")

# case_id -> list of acceptable pt_codes (ints, verified from DB 2026-05-26)
# These are case-specific -- not universal synonyms for the PT name.
# Rationale per case is documented below.
ACCEPTABLE = {
    # Hyperglycaemia cases where report describes BG elevation numerically
    # without using the word "hyperglycaemia/hyperglycemia"
    # -> "Blood glucose increased" (10005557) is a plausible alternative PT
    "3004464228-2024-03085": [10005557],   # BG 600+ mg/dL, cannula dislodged
    "3013756811-2024-09995": [10005557],   # BG 700 mg/dL, ICU admission
    "3013756811-2024-17732": [10005557],   # BG 212 mg/dL, fill-notification report
    "3004464228-2024-27443": [10005557],   # BG 250+ mg/dL, cannula dislodged

    # Application site haematoma: report uses "bleeding" / "haemorrhage" language
    # -> "Infusion site haemorrhage" (10065464) plausible for infusion-pump reports
    "3004464228-2024-04816": [10065464],   # Infusion site bleeding, cannula bent

    # Application site haematoma: device-alarm + hemorrhage report, no site-specific detail
    # -> "Medical device site haemorrhage" (10075578) plausible
    "2032227-2024-227041":   [10075578],   # No-delivery alarm + hemorrhage/bleeding

    # Application site dermatitis: report describes skin irritation / allergic reaction
    # at device application site -- dermatitis-spectrum PTs are plausible alternatives
    "3004464228-2024-04262": [10075572, 10040914],  # Medical device site dermatitis, Skin reaction
    "3004464228-2024-29565": [10075572, 10040880],  # Medical device site dermatitis, Skin irritation
}


def main():
    if not GOLDEN_SET.exists():
        raise FileNotFoundError(f"Not found: {GOLDEN_SET}")

    # Backup
    shutil.copy(GOLDEN_SET, BACKUP)
    print(f"Backup written: {BACKUP}")

    cases = [json.loads(line) for line in GOLDEN_SET.read_text().splitlines() if line.strip()]

    patched = 0
    for case in cases:
        key = case["mdr_report_key"]
        if key in ACCEPTABLE:
            case["acceptable_pt_codes"] = ACCEPTABLE[key]
            patched += 1

    GOLDEN_SET.write_text("\n".join(json.dumps(c) for c in cases) + "\n")
    print(f"Patched {patched} cases (of {len(cases)} total).")
    print(f"Written: {GOLDEN_SET}")


if __name__ == "__main__":
    main()
