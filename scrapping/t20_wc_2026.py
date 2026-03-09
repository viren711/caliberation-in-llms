"""
T20 World Cup 2026 - Match Results Scraper
==========================================
Scrapes all match results (teams, winner, date, venue) from ESPNcricinfo.

Sources:
  - Primary  : ESPNcricinfo series page (HTML scraping via BeautifulSoup)
  - Fallback : Wikipedia infobox table (if ESPNcricinfo blocks the request)

Requirements:
    pip install requests beautifulsoup4 lxml tabulate
"""

import re
import time
import requests
from bs4 import BeautifulSoup
from tabulate import tabulate

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

ESPN_URL = (
    "https://www.espncricinfo.com/series/"
    "icc-men-s-t20-world-cup-2025-26-1502138/match-schedule-fixtures-and-results"
)

WIKI_URL = "https://en.wikipedia.org/wiki/2026_Men%27s_T20_World_Cup"


# ─────────────────────────────────────────────
# SOURCE 1 — ESPNcricinfo
# ─────────────────────────────────────────────
def scrape_espncricinfo() -> list[dict]:
    """
    Scrapes match cards from the ESPNcricinfo series schedule page.
    Returns a list of dicts: {match, date, venue, teams, result, winner}
    """
    print("⏳ Fetching from ESPNcricinfo …")
    resp = requests.get(ESPN_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    matches = []

    # Each match is inside a <div> with class containing 'match-info'
    cards = soup.find_all("div", class_=re.compile(r"match-info|MatchInfo"))

    for card in cards:
        try:
            # Match number / label
            label_tag = card.find(class_=re.compile(r"match-no|MatchNo|series-name|SeriesName", re.I))
            label = label_tag.get_text(strip=True) if label_tag else ""

            # Date
            date_tag = card.find(class_=re.compile(r"match-date|MatchDate|date", re.I))
            date = date_tag.get_text(strip=True) if date_tag else ""

            # Venue
            venue_tag = card.find(class_=re.compile(r"ground|venue|stadium", re.I))
            venue = venue_tag.get_text(strip=True) if venue_tag else ""

            # Teams
            team_tags = card.find_all(class_=re.compile(r"team-name|TeamName|team", re.I))
            teams = " vs ".join(t.get_text(strip=True) for t in team_tags[:2])

            # Result
            result_tag = card.find(class_=re.compile(r"status|result|match-status", re.I))
            result = result_tag.get_text(strip=True) if result_tag else "Upcoming / No result"

            # Winner (first word(s) before "won")
            winner = ""
            m = re.search(r"^(.+?)\s+won", result, re.IGNORECASE)
            if m:
                winner = m.group(1).strip()
            elif "no result" in result.lower():
                winner = "No Result"
            elif "upcoming" in result.lower() or result == "":
                winner = "TBD"

            matches.append({
                "Match":  label or f"Match {len(matches)+1}",
                "Date":   date,
                "Venue":  venue,
                "Teams":  teams,
                "Result": result,
                "Winner": winner,
            })
        except Exception:
            continue

    return matches


# ─────────────────────────────────────────────
# SOURCE 2 — Wikipedia (fallback / supplement)
# ─────────────────────────────────────────────
def scrape_wikipedia() -> list[dict]:
    """
    Scrapes the 2026 T20 World Cup Wikipedia page for match result tables.
    Returns a list of dicts: {match, date, teams, result, winner}
    """
    print("⏳ Fetching from Wikipedia …")
    resp = requests.get(WIKI_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    matches = []
    match_num = 1

    # Wikipedia stores match summaries inside wikitables
    tables = soup.find_all("table", class_="wikitable")

    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) < 3:
                continue

            text = " | ".join(c.get_text(" ", strip=True) for c in cells)

            # Skip header rows
            if re.search(r"match|date|team|result", text, re.I) and len(cells) < 5:
                continue

            # Look for rows containing "won" or "no result"
            if not re.search(r"\bwon\b|no result|abandoned", text, re.I):
                continue

            # Try to extract date (e.g. "7 February")
            date_m = re.search(
                r"\b(\d{1,2}\s+(?:January|February|March)(?:\s+\d{4})?)\b",
                text, re.I
            )
            date = date_m.group(1) if date_m else ""

            # Extract result sentence
            result_m = re.search(r"([A-Za-z ]+(?:won|No result|abandoned)[^|]*)", text)
            result = result_m.group(1).strip() if result_m else text[:120]

            winner_m = re.search(r"^(.+?)\s+won", result, re.IGNORECASE)
            winner = winner_m.group(1).strip() if winner_m else (
                "No Result" if re.search(r"no result|abandoned", result, re.I) else ""
            )

            matches.append({
                "Match":  f"Match {match_num}",
                "Date":   date,
                "Venue":  "",
                "Teams":  "",
                "Result": result,
                "Winner": winner,
            })
            match_num += 1
            time.sleep(0.05)

    return matches


# ─────────────────────────────────────────────
# SOURCE 3 — Cricbuzz API-style endpoint
# ─────────────────────────────────────────────
CRICBUZZ_URL = (
    "https://www.cricbuzz.com/cricket-series/7476/"
    "icc-mens-t20-world-cup-2026/matches"
)

def scrape_cricbuzz() -> list[dict]:
    """
    Scrapes Cricbuzz series matches page for T20 WC 2026.
    Returns a list of dicts.
    """
    print("⏳ Fetching from Cricbuzz …")
    resp = requests.get(CRICBUZZ_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    matches = []

    # Cricbuzz wraps each match in a div.cb-col.cb-col-100.cb-lst-itm
    cards = soup.find_all("div", class_=re.compile(r"cb-lst-itm|cb-series-matches", re.I))

    for card in cards:
        try:
            # Match title (e.g. "1st Match, Group A")
            title_tag = card.find(class_=re.compile(r"cb-series-brdr|match-hdr|cb-mtch-lst", re.I))
            match_title = title_tag.get_text(strip=True) if title_tag else ""

            # Date + venue usually in a single line
            meta_tag = card.find(class_=re.compile(r"schedule-date|cb-font-12", re.I))
            meta = meta_tag.get_text(" ", strip=True) if meta_tag else ""

            # Teams
            team_tags = card.find_all(class_=re.compile(r"cb-ovr-flo|cb-team", re.I))
            teams = " vs ".join(t.get_text(strip=True) for t in team_tags[:2])

            # Result
            result_tag = card.find(class_=re.compile(r"cb-text-complete|cb-text-live|cb-text-preview", re.I))
            result = result_tag.get_text(strip=True) if result_tag else "Upcoming"

            winner_m = re.search(r"^(.+?)\s+won", result, re.IGNORECASE)
            winner = winner_m.group(1).strip() if winner_m else (
                "No Result" if "no result" in result.lower() else "TBD"
            )

            matches.append({
                "Match":  match_title or f"Match {len(matches)+1}",
                "Date":   meta,
                "Venue":  "",
                "Teams":  teams,
                "Result": result,
                "Winner": winner,
            })
        except Exception:
            continue

    return matches


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  🏏  T20 World Cup 2026 — Match Results Scraper")
    print("=" * 60)

    matches = []

    # Try sources in order of preference
    for scraper, name in [
        (scrape_espncricinfo, "ESPNcricinfo"),
        (scrape_cricbuzz,     "Cricbuzz"),
        (scrape_wikipedia,    "Wikipedia"),
    ]:
        try:
            results = scraper()
            if results:
                print(f"✅ Got {len(results)} matches from {name}\n")
                matches = results
                break
            else:
                print(f"⚠️  {name} returned 0 matches, trying next source …")
        except Exception as e:
            print(f"❌ {name} failed: {e}, trying next source …")

    if not matches:
        print("❌ All sources failed. Check your internet connection.")
        return

    # ── Print table ──────────────────────────────────────
    display = [
        {
            "#":       i + 1,
            "Match":   m["Match"],
            "Date":    m["Date"],
            "Teams":   m["Teams"],
            "Winner":  m["Winner"],
            "Result":  m["Result"][:60] + ("…" if len(m["Result"]) > 60 else ""),
        }
        for i, m in enumerate(matches)
    ]

    print(tabulate(display, headers="keys", tablefmt="rounded_outline"))

    # ── Save to CSV ──────────────────────────────────────
    import csv, os
    out_path = os.path.join(os.path.dirname(__file__) or ".", "t20_wc_2026_results.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Match", "Date", "Venue", "Teams", "Result", "Winner"])
        writer.writeheader()
        writer.writerows(matches)

    print(f"\n💾 Results saved to: {out_path}")
    print(f"📊 Total matches scraped: {len(matches)}")


if __name__ == "__main__":
    main()