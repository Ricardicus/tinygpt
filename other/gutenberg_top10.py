#!/usr/bin/env python3
"""
Download the 10 most-downloaded books from Project Gutenberg
(Top 100 EBooks last 7 days) as plain text, and save them with
sane filenames in ./gutenberg_top10.

Requirements:
    pip install requests beautifulsoup4
"""

import os
import re
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.gutenberg.org"
TOP_URL = f"{BASE_URL}/browse/scores/top"

# Be nice: identify yourself and add a small delay between downloads.
HEADERS = {
    "User-Agent": (
        "GutenbergTop10Script/1.0 (https://example.com; "
        "contact: your_email@example.com)"
    )
}

DOWNLOAD_DIR = "gutenberg_top10"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def slugify(text: str, maxlen: int = 80) -> str:
    """Turn a title into a filesystem-friendly slug."""
    text = re.sub(r"\s+", " ", text).strip()
    # Keep letters, numbers, spaces, dashes and underscores
    text = re.sub(r"[^A-Za-z0-9 _\-]+", "", text)
    text = text.replace(" ", "_")
    if not text:
        text = "book"
    return text[:maxlen]


def get_top10_ebook_links():
    """Return a list of (rank, title, url) tuples for the top 10 ebooks."""
    print(f"Fetching top list from {TOP_URL} ...")
    resp = requests.get(TOP_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Find the "Top 100 EBooks last 7 days" section, then the following <ol>
    header = soup.find(
        lambda tag: tag.name in ("h2", "h3")
        and "Top 100 EBooks last 7 days" in tag.get_text()
    )
    if header is None:
        raise RuntimeError("Could not find 'Top 100 EBooks last 7 days' section.")

    ol = header.find_next("ol")
    if ol is None:
        raise RuntimeError("Could not find the list (<ol>) for top ebooks.")

    links = []
    for rank, li in enumerate(ol.find_all("li", recursive=False), start=1):
        if rank > 10:
            break
        a = li.find("a", href=True)
        if not a:
            continue
        title = a.get_text(strip=True)
        url = urljoin(BASE_URL, a["href"])
        links.append((rank, title, url))

    if len(links) == 0:
        raise RuntimeError("No ebook links found in the top list.")

    return links


def find_plain_text_link(ebook_page_html: str) -> str | None:
    """Find the 'Plain Text' download link on an ebook page, if possible."""
    soup = BeautifulSoup(ebook_page_html, "html.parser")

    # Prefer an explicit "Plain Text" entry in the "Read or download" section
    for a in soup.find_all("a", href=True):
        text = a.get_text(" ", strip=True)
        href = a["href"]
        if "Plain Text" in text and ".txt" in href:
            return urljoin(BASE_URL, href)

    # Fallback: any .txt download
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if ".txt" in href:
            return urljoin(BASE_URL, href)

    return None


def download_book(rank: int, title: str, ebook_url: str):
    print(f"[{rank:02d}] Fetching ebook page: {ebook_url}")
    resp = requests.get(ebook_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    text_url = find_plain_text_link(resp.text)
    if not text_url:
        print(f"  -> No plain text link found for '{title}'. Skipping.")
        return

    print(f"  -> Downloading plain text: {text_url}")
    book_resp = requests.get(text_url, headers=HEADERS, timeout=60)
    book_resp.raise_for_status()

    # Build a nice filename: "01_Frankenstein_Or_The_Modern_Prometheus_by_Mary_Wollstonecraft_Shelley.txt"
    safe_title = slugify(title)
    filename = f"{rank:02d}_{safe_title}.txt"
    filepath = os.path.join(DOWNLOAD_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(book_resp.content)

    print(f"  -> Saved to {filepath}")


def main():
    top_books = get_top10_ebook_links()
    print(f"Found {len(top_books)} top titles. Downloading the first 10...\n")

    for rank, title, url in top_books:
        download_book(rank, title, url)
        # Be polite to the server
        time.sleep(5)

    print("\nDone! Files are in:", os.path.abspath(DOWNLOAD_DIR))


if __name__ == "__main__":
    main()

