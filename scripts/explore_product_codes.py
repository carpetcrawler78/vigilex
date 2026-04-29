"""
explore_product_codes.py -- map device categories to FDA product codes.

For Block E we need MAUDE adverse event data spread across multiple
product_code values so PRR/ROR has comparison populations. This script
queries openFDA and shows, for each device family we care about:

  - the top FDA product codes (3-letter codes)
  - how many MAUDE reports exist per code (proxy for ingest volume)
  - one example device.generic_name + device.brand_name per code
    so we can sanity-check what the code actually represents

Run:
  python scripts/explore_product_codes.py            # no API key
  python scripts/explore_product_codes.py --key KEY  # higher rate limit

Output is printed as plain text tables -- pick the codes you want to
ingest, then we extend flatten_maude_record() and re-ingest.
"""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from typing import Optional

API_BASE = "https://api.fda.gov/device/event.json"

# Device families we want to cover for the demo.
# Search terms hit device.generic_name (case-insensitive substring match).
CATEGORIES: dict[str, list[str]] = {
    "Insulin pump":              ["insulin infusion pump", "insulin pump"],
    "Continuous glucose monitor": ["continuous glucose monitor", "glucose sensor"],
    "Blood glucose monitor":      ["blood glucose monitor", "glucose meter"],
    "Insulin pen":                ["insulin pen", "insulin injector"],
    "Pacemaker":                  ["pacemaker pulse generator", "pacemaker"],
    "ICD":                        ["implantable cardioverter defibrillator", "defibrillator implantable"],
    "Pacing lead":                ["pacemaker lead", "pacing lead"],
}

TOP_N_CODES = 5      # how many product codes to show per family
EXAMPLE_RECORDS = 1  # how many sample records to fetch per code for naming


def fetch(url: str, retries: int = 3) -> Optional[dict]:
    """GET a URL, return parsed JSON or None on failure."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [warn] fetch failed: {e}", file=sys.stderr)
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def top_codes_for_search(search_term: str, api_key: str) -> list[tuple[str, int]]:
    """Return [(product_code, count), ...] for adverse events matching the term."""
    params = {
        "search": f'device.generic_name:"{search_term}"',
        "count":  "device.device_report_product_code.exact",
        "limit":  str(TOP_N_CODES * 3),  # over-fetch, we filter dupes
    }
    if api_key:
        params["api_key"] = api_key
    url = f"{API_BASE}?{urllib.parse.urlencode(params)}"
    data = fetch(url)
    if not data or "results" not in data:
        return []
    return [(r["term"], r["count"]) for r in data["results"]]


def example_device_for_code(product_code: str, api_key: str) -> tuple[str, str]:
    """Return (generic_name, brand_name) for one example record with this code."""
    params = {
        "search": f'device.device_report_product_code:"{product_code}"',
        "limit":  str(EXAMPLE_RECORDS),
    }
    if api_key:
        params["api_key"] = api_key
    url = f"{API_BASE}?{urllib.parse.urlencode(params)}"
    data = fetch(url)
    if not data or "results" not in data:
        return ("?", "?")
    rec = data["results"][0]
    devs = rec.get("device") or [{}]
    dev = devs[0] if devs else {}
    return (dev.get("generic_name", "?") or "?",
            dev.get("brand_name",   "?") or "?")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="", help="openFDA API key (optional)")
    args = ap.parse_args()

    print("=" * 78)
    print("openFDA MAUDE -- product code reconnaissance")
    print("=" * 78)

    for family, search_terms in CATEGORIES.items():
        print(f"\n## {family}")
        seen: dict[str, int] = {}
        for term in search_terms:
            for code, count in top_codes_for_search(term, args.key):
                seen[code] = max(seen.get(code, 0), count)
            time.sleep(0.3)  # be polite to the API

        if not seen:
            print("  (no matches)")
            continue

        ranked = sorted(seen.items(), key=lambda x: -x[1])[:TOP_N_CODES]

        print(f"  {'CODE':<6} {'REPORTS':>10}  GENERIC NAME / BRAND (example)")
        print(f"  {'----':<6} {'-------':>10}  -------------------------------")
        for code, count in ranked:
            generic, brand = example_device_for_code(code, args.key)
            generic = (generic[:35] + "..") if len(generic) > 37 else generic
            brand   = (brand[:25]   + "..") if len(brand)   > 27 else brand
            print(f"  {code:<6} {count:>10,}  {generic:<37}  / {brand}")
            time.sleep(0.3)

    print("\n" + "=" * 78)
    print("Done. Pick a final mix of codes; we then re-ingest with event_type")
    print("and patient_outcomes captured.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
