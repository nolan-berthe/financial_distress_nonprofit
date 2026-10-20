"""Pull Form 990 financials by EIN from the ProPublica Nonprofit Explorer API.

Polite by construction: every response is cached to disk by EIN forever (nonprofit
990 history for a past fiscal year does not change), and a fixed delay is inserted
between live network requests so repeated runs only pay the rate-limit cost once.
"""

import json
import logging
import time

import requests

from config import FORM990_CACHE_DIR, PROPUBLICA_ORG_URL, PROPUBLICA_RATE_LIMIT_SECONDS

logger = logging.getLogger(__name__)


def _cache_path(ein: str):
    return FORM990_CACHE_DIR / f"{ein}.json"


def get_organization(ein: str, force: bool = False) -> dict | None:
    """Fetch (or load cached) ProPublica org record for a 9-digit EIN string."""
    ein = str(ein).strip().zfill(9)
    cache_path = _cache_path(ein)

    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    url = PROPUBLICA_ORG_URL.format(ein=ein)
    try:
        resp = requests.get(url, timeout=30)
    except requests.RequestException:
        logger.exception("Network error fetching EIN %s", ein)
        return None

    time.sleep(PROPUBLICA_RATE_LIMIT_SECONDS)

    if resp.status_code == 404:
        logger.info("EIN %s not found in ProPublica Nonprofit Explorer", ein)
        data = {"ein": ein, "not_found": True}
    elif resp.status_code != 200:
        logger.warning("EIN %s fetch failed: HTTP %s", ein, resp.status_code)
        return None
    else:
        data = resp.json()

    cache_path.write_text(json.dumps(data), encoding="utf-8")
    return data


def fetch_many(eins, force: bool = False, progress_every: int = 50) -> dict:
    """Fetch/cache a batch of EINs. Returns {ein: org_record_or_None}."""
    results = {}
    unique_eins = sorted(set(str(e).strip().zfill(9) for e in eins if str(e).strip() not in ("", "nan", "000000000")))
    for i, ein in enumerate(unique_eins, 1):
        results[ein] = get_organization(ein, force=force)
        if i % progress_every == 0:
            logger.info("Fetched %d/%d EINs", i, len(unique_eins))
    return results


def extract_filing_for_year(org_record: dict | None, tax_year: int) -> dict:
    """Pull one fiscal year's financials out of a cached org record.

    `tax_year` is matched against the filing's `tax_prd_yr` (the calendar year the
    organization's fiscal period ended in), which lines up with how `fiscal_year` is
    defined for the IPEDS/composite-score panel.
    """
    empty = {
        "f990_total_revenue": None,
        "f990_total_expenses": None,
        "f990_total_assets_end": None,
        "f990_total_liabilities_end": None,
        "f990_net_assets_end": None,
    }
    if not org_record or org_record.get("not_found"):
        return empty

    for filing in org_record.get("filings_with_data", []) or []:
        if filing.get("tax_prd_yr") == tax_year:
            return {
                "f990_total_revenue": filing.get("totrevenue"),
                "f990_total_expenses": filing.get("totfuncexpns"),
                "f990_total_assets_end": filing.get("totassetsend"),
                "f990_total_liabilities_end": filing.get("totliabend"),
                "f990_net_assets_end": filing.get("totnetassetend"),
            }
    return empty


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    test_ein = sys.argv[1] if len(sys.argv) > 1 else "630366652"
    rec = get_organization(test_ein)
    print(json.dumps(rec, indent=2)[:2000])
