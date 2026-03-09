"""
T20 World Cup 2026 — Parse from Saved Webpage
==============================================
Steps:
  1. Open the match results page in your browser
     (ESPNcricinfo / Cricbuzz / Wikipedia / ICC website)
  2. Press Ctrl+S (or Cmd+S on Mac) → save as "Webpage, Complete" or "Webpage, HTML Only"
  3. Run:  python parse_t20_html.py your_saved_file.html

Output:
  - Prints a formatted table in the terminal
  - Saves t20_wc_2026_results.csv in the same folder

Install dependencies:
    pip install beautifulsoup4 lxml tabulate
"""

import re
import sys
import csv
from pathlib import Path
from bs4 import BeautifulSoup
from tabulate import tabulate


# ── Load file ────────────────────────────────────────────────────
def load_html(path: str) -> BeautifulSoup:
    file = Path(path)
    if not file.exists():
        print(f"❌ File not found: {path}")
        sys.exit(1)
    print(f"✅ Loaded: {file.name}  ({file.stat().st_size // 1024} KB)")
    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        return BeautifulSoup(f.read(), "lxml")


# ── Detect which site the HTML is from ──────────────────────────
def detect_source(soup: BeautifulSoup) -> str:
    title = soup.title.string.lower() if soup.title else ""
    text  = soup.get_text()[:500].lower()
    if "espncricinfo" in title or "espncricinfo" in text:
        return "espncricinfo"
    if "cricbuzz" in title or "cricbuzz" in text:
        return "cricbuzz"
    if "wikipedia" in title or "wikimedia" in text:
        return "wikipedia"
    if "icc-cricket" in title or "icc" in title:
        return "icc"
    return "generic"


# ══════════════════════════════════════════════════════════════════
# PARSERS — one per site
# ══════════════════════════════════════════════════════════════════

def parse_espncricinfo(soup: BeautifulSoup) -> list[dict]:
    matches = []
    # Each match card sits inside an <a> or <div> with these class patterns
    cards = soup.find_all(
        lambda tag: tag.name in ("div", "article")
        and any(
            kw in " ".join(tag.get("class", []))
            for kw in ("match-info", "MatchInfo", "match-card", "ds-border")
        )
    )

    for i, card in enumerate(cards, 1):
        text = card.get_text(" ", strip=True)

        # Date — e.g. "Sat, 7 Feb" or "07 Feb 2026"
        date_m = re.search(
            r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)?,?\s*"
            r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*"
            r"(?:\s+\d{4})?)", text, re.I
        )
        date = date_m.group(1) if date_m else ""

        # Teams — look for "Team1 vs Team2" pattern
        teams_m = re.search(r"([A-Z][a-zA-Z ]+?)\s+[vV][sS]\.?\s+([A-Z][a-zA-Z ]+)", text)
        teams = f"{teams_m.group(1).strip()} vs {teams_m.group(2).strip()}" if teams_m else ""

        # Result sentence
        result_m = re.search(
            r"([A-Za-z ]+(?:won|No result|abandoned|tied)[^\n]{0,80})",
            text, re.I
        )
        result = result_m.group(1).strip() if result_m else "Upcoming / TBD"

        # Winner
        winner_m = re.search(r"^(.+?)\s+won", result, re.I)
        winner = (
            winner_m.group(1).strip() if winner_m
            else ("No Result" if re.search(r"no result|abandon", result, re.I) else "TBD")
        )

        # Venue
        venue_m = re.search(r"(?:at|@)\s+([A-Z][^,\n]{5,50})", text)
        venue = venue_m.group(1).strip() if venue_m else ""

        matches.append({
            "Match":  f"Match {i}",
            "Date":   date,
            "Venue":  venue,
            "Teams":  teams,
            "Result": result,
            "Winner": winner,
        })

    return matches


def parse_cricbuzz(soup: BeautifulSoup) -> list[dict]:
    matches = []
    # Cricbuzz match rows: div.cb-series-matches or div.cb-lst-itm
    cards = soup.find_all(
        "div",
        class_=re.compile(r"cb-lst-itm|cb-series-matches|cb-mtch-lst", re.I)
    )

    for i, card in enumerate(cards, 1):
        text = card.get_text(" ", strip=True)

        date_m = re.search(
            r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*"
            r"(?:,?\s*\d{4})?)\b", text, re.I
        )
        date = date_m.group(1) if date_m else ""

        # Match label e.g. "1st Match, Group A"
        label_m = re.search(r"(\d+(?:st|nd|rd|th)\s+Match[^,\n]*)", text, re.I)
        label = label_m.group(1) if label_m else f"Match {i}"

        teams_m = re.search(r"([A-Z][a-zA-Z ]+?)\s+vs\s+([A-Z][a-zA-Z ]+)", text)
        teams = f"{teams_m.group(1).strip()} vs {teams_m.group(2).strip()}" if teams_m else ""

        result_m = re.search(
            r"([A-Za-z ]+(?:won|No result|abandoned|tied)[^\n]{0,80})",
            text, re.I
        )
        result = result_m.group(1).strip() if result_m else "Upcoming / TBD"

        winner_m = re.search(r"^(.+?)\s+won", result, re.I)
        winner = (
            winner_m.group(1).strip() if winner_m
            else ("No Result" if re.search(r"no result|abandon", result, re.I) else "TBD")
        )

        matches.append({
            "Match":  label,
            "Date":   date,
            "Venue":  "",
            "Teams":  teams,
            "Result": result,
            "Winner": winner,
        })

    return matches


def parse_wikipedia(soup: BeautifulSoup) -> list[dict]:
    matches = []
    match_num = 1

    for table in soup.find_all("table", class_=re.compile("wikitable", re.I)):
        for row in table.find_all("tr")[1:]:
            cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
            text = " | ".join(cells)

            if not re.search(r"\bwon\b|no result|abandoned|tied", text, re.I):
                continue

            date_m = re.search(
                r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*"
                r"(?:\s+\d{4})?)\b", text, re.I
            )
            date = date_m.group(1) if date_m else ""

            result_m = re.search(
                r"([A-Za-z ]+(?:won|No result|abandoned|tied)[^\|]{0,80})",
                text, re.I
            )
            result = result_m.group(1).strip() if result_m else ""

            winner_m = re.search(r"^(.+?)\s+won", result, re.I)
            winner = (
                winner_m.group(1).strip() if winner_m
                else ("No Result" if re.search(r"no result|abandon", result, re.I) else "")
            )

            teams_m = re.search(r"([A-Z][a-zA-Z ]+?)\s+(?:v|vs)\.?\s+([A-Z][a-zA-Z ]+)", text)
            teams = f"{teams_m.group(1).strip()} vs {teams_m.group(2).strip()}" if teams_m else ""

            matches.append({
                "Match":  f"Match {match_num}",
                "Date":   date,
                "Venue":  "",
                "Teams":  teams,
                "Result": result,
                "Winner": winner,
            })
            match_num += 1

    return matches


def parse_icc(soup: BeautifulSoup) -> list[dict]:
    """Works for icc-cricket.com saved pages."""
    matches = []
    cards = soup.find_all(
        lambda tag: tag.name in ("div", "article", "li")
        and any(
            kw in " ".join(tag.get("class", []))
            for kw in ("match", "fixture", "result", "card")
        )
    )

    for i, card in enumerate(cards, 1):
        text = card.get_text(" ", strip=True)
        if len(text) < 10:
            continue

        date_m = re.search(
            r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*"
            r"(?:\s+\d{4})?)\b", text, re.I
        )
        date = date_m.group(1) if date_m else ""

        teams_m = re.search(r"([A-Z][a-zA-Z ]+?)\s+vs\.?\s+([A-Z][a-zA-Z ]+)", text)
        teams = f"{teams_m.group(1).strip()} vs {teams_m.group(2).strip()}" if teams_m else ""

        result_m = re.search(
            r"([A-Za-z ]+(?:won|No result|abandoned|tied)[^\n]{0,80})",
            text, re.I
        )
        result = result_m.group(1).strip() if result_m else "Upcoming / TBD"

        winner_m = re.search(r"^(.+?)\s+won", result, re.I)
        winner = (
            winner_m.group(1).strip() if winner_m
            else ("No Result" if re.search(r"no result|abandon", result, re.I) else "TBD")
        )

        matches.append({
            "Match":  f"Match {i}",
            "Date":   date,
            "Venue":  "",
            "Teams":  teams,
            "Result": result,
            "Winner": winner,
        })

    return matches


def parse_generic(soup: BeautifulSoup) -> list[dict]:
    """
    Last-resort parser: scans ALL text in the page for lines/sentences
    that contain 'won', 'no result', etc.
    """
    matches = []
    full_text = soup.get_text("\n", strip=True)
    lines = full_text.splitlines()
    match_num = 1

    for line in lines:
        line = line.strip()
        if not re.search(r"\bwon\b|no result|abandoned|tied", line, re.I):
            continue
        if len(line) < 10 or len(line) > 300:
            continue

        date_m = re.search(
            r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*"
            r"(?:\s+\d{4})?)\b", line, re.I
        )
        date = date_m.group(1) if date_m else ""

        winner_m = re.search(r"^(.+?)\s+won", line, re.I)
        winner = (
            winner_m.group(1).strip() if winner_m
            else ("No Result" if re.search(r"no result|abandon", line, re.I) else "")
        )

        teams_m = re.search(r"([A-Z][a-zA-Z ]+?)\s+vs\.?\s+([A-Z][a-zA-Z ]+)", line)
        teams = f"{teams_m.group(1).strip()} vs {teams_m.group(2).strip()}" if teams_m else ""

        matches.append({
            "Match":  f"Match {match_num}",
            "Date":   date,
            "Venue":  "",
            "Teams":  teams,
            "Result": line[:120],
            "Winner": winner,
        })
        match_num += 1

    return matches


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Usage: python parse_t20_html.py <saved_webpage.html>")
        sys.exit(0)

    html_path = sys.argv[1]
    soup = load_html(html_path)

    source = detect_source(soup)
    print(f"🔍 Detected source: {source.upper()}")

    parser_map = {
        "espncricinfo": parse_espncricinfo,
        "cricbuzz":     parse_cricbuzz,
        "wikipedia":    parse_wikipedia,
        "icc":          parse_icc,
        "generic":      parse_generic,
    }

    matches = parser_map[source](soup)

    # If site-specific parser got nothing, fall back to generic
    if not matches:
        print("⚠️  Site-specific parser found 0 matches, trying generic parser …")
        matches = parse_generic(soup)

    if not matches:
        print("❌ No match data found. Try saving a different page (match results/schedule page).")
        sys.exit(1)

    # ── Print table ──────────────────────────────────────────────
    print(f"\n✅ Found {len(matches)} matches\n")
    display = [
        {
            "#":      i + 1,
            "Match":  m["Match"],
            "Date":   m["Date"],
            "Teams":  m["Teams"],
            "Winner": m["Winner"],
            "Result": (m["Result"][:55] + "…") if len(m["Result"]) > 55 else m["Result"],
        }
        for i, m in enumerate(matches)
    ]
    print(tabulate(display, headers="keys", tablefmt="rounded_outline"))

    # ── Save CSV ─────────────────────────────────────────────────
    out = Path(html_path).with_name("t20_wc_2026_results.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Match", "Date", "Venue", "Teams", "Result", "Winner"]
        )
        writer.writeheader()
        writer.writerows(matches)

    print(f"\n💾 CSV saved to: {out}")


if __name__ == "__main__":
    main()