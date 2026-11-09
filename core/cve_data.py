"""CVE descriptions + CVSS severity from the NVD API, for the severity classifier.

Downloads every CVE published in a date window (default Jan 1 - Apr 29 2024,
~12.5k CVEs) from the public NVD CVE API 2.0 and keeps only
    id, published date, English description, CVSS v3.1 base score / severity
in data/nvd_cves_2024.parquet. Label: 1 = HIGH or CRITICAL (score >= 7.0), 0 = LOW/MEDIUM.

Split: by publication date (first 80% train, last 20% test), so the model is
tested on CVEs published after the ones it learned from, like it would be used.
"""
import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
PARQUET = os.path.join(DATA_DIR, "nvd_cves_2024.parquet")
API = "https://services.nvd.nist.gov/rest/json/cves/2.0"
WINDOW = ("2024-01-01T00:00:00.000", "2024-04-29T23:59:59.999")  # NVD max window: 120 days
# Untouched confirmatory set: CVEs published after the train/test window, never used for design.
HOLDOUT_WINDOW = ("2024-05-01T00:00:00.000", "2024-08-28T23:59:59.999")
HOLDOUT_PARQUET = os.path.join(DATA_DIR, "nvd_cves_2024_holdout.parquet")
PAGE = 2000


def _get(params):
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "quantum-agi-prototype"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception as e:  # NVD rate-limits (403/503); back off and retry
            wait = 10 * (attempt + 1)
            print(f"  NVD request failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("NVD API kept failing; try again later")


def _parse(item):
    cve = item["cve"]
    if cve.get("vulnStatus") == "Rejected":
        return None
    metrics = cve.get("metrics", {}).get("cvssMetricV31", [])
    if not metrics:
        return None
    m = next((m for m in metrics if m.get("type") == "Primary"), metrics[0])["cvssData"]
    desc = next((d["value"] for d in cve.get("descriptions", []) if d["lang"] == "en"), None)
    if not desc:
        return None
    return {"id": cve["id"], "published": cve["published"], "description": desc,
            "score": float(m["baseScore"]), "severity": m["baseSeverity"]}


def download(window=WINDOW, path=PARQUET):
    import pandas as pd
    os.makedirs(DATA_DIR, exist_ok=True)
    rows, start, total = [], 0, None
    while total is None or start < total:
        d = _get({"pubStartDate": window[0], "pubEndDate": window[1],
                  "resultsPerPage": PAGE, "startIndex": start})
        total = d["totalResults"]
        rows += [r for r in map(_parse, d["vulnerabilities"]) if r]
        start += PAGE
        print(f"  fetched {min(start, total)}/{total} CVEs")
        if start < total:
            time.sleep(6)  # NVD asks for <= 5 requests / 30 s without an API key
    df = pd.DataFrame(rows).sort_values("published").reset_index(drop=True)
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} CVEs with CVSS v3.1 scores to {path}")
    return df


def load_cves():
    import pandas as pd
    if not os.path.exists(PARQUET):
        print("Downloading CVEs from NVD (one-off, ~53 MB over ~1 minute)...")
        return download()
    return pd.read_parquet(PARQUET)


def load_cve_split(split="train"):
    """Descriptions and labels (1 = HIGH/CRITICAL) for the time-based split.

    split = "train" | "test" (80/20 by date, Jan-Apr 2024) | "holdout" (May-Aug 2024,
    the confirmatory set; see results/confirmatory/PROTOCOL.md)."""
    if split == "holdout":
        import pandas as pd
        df = pd.read_parquet(HOLDOUT_PARQUET) if os.path.exists(HOLDOUT_PARQUET) else download(HOLDOUT_WINDOW, HOLDOUT_PARQUET)
        return df["description"].tolist(), (df["score"].to_numpy() >= 7.0).astype(int)
    df = load_cves()
    cut = int(len(df) * 0.8)
    part = df.iloc[:cut] if split == "train" else df.iloc[cut:]
    return part["description"].tolist(), (part["score"].to_numpy() >= 7.0).astype(int)
