"""Part 1: crawl www.zaio.io and save the cleaned text of every page.

    python -m app.crawl                  # crawl, clean, write data/website/pages.json
    python -m app.crawl --max-pages 10   # quick look at the first few pages

Pages are opened in headless Chromium through Playwright (the Python
equivalent of Puppeteer), so anything the site builds with JavaScript is
there when the text is read. The cleaned result is saved to a JSON file and
committed, so `python -m app.ingest` and the tests don't need the network,
and the answers always match a known snapshot of the site.
"""

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from app import config

USER_AGENT = "Mozilla/5.0 (compatible; ZaioStudentAssistant/2.0; student RAG project)"

# Site chrome. The Next.js pages use proper <header>/<footer> elements, so
# these catch the menu, the footer link lists and the social icons.
CHROME = ["header", "footer", "nav", "aside", "dialog",
          "[role=navigation]", "[role=banner]", "[role=contentinfo]", "[class*=ookie]"]
# Never text: code, styling, icons, embedded players, forms' option lists.
NOT_TEXT = ["script", "style", "noscript", "template", "svg", "iframe", "video", "select", "img"]

HEADINGS = ["h1", "h2", "h3", "h4"]
# Tags that start a new line. Anything else (span, a, strong...) is inline,
# so "Pay <strong>R 38,950</strong> upfront" stays on one line.
BLOCKS = ["address", "article", "blockquote", "br", "button", "dd", "details", "div", "dl", "dt",
          "figcaption", "h1", "h2", "h3", "h4", "h5", "h6", "hr", "li", "main", "ol", "p",
          "pre", "section", "summary", "table", "td", "th", "tr", "ul"]
HEADING_MARK = "\x01"  # tags heading lines while the tree is flattened to text

# Links to these are files, not pages.
FILE_EXTENSIONS = re.compile(r"\.(pdf|jpe?g|png|gif|webp|svg|ico|mp4|zip|docx?|xlsx?|css|js|json|xml)$", re.I)


@dataclass
class Section:
    heading: str
    text: str


@dataclass
class WebPage:
    url: str
    title: str
    sections: list[Section] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(s.text for s in self.sections)


# --- robots.txt -------------------------------------------------------------

class RobotsRules:
    """Allow/Disallow rules for one user agent, matched the way RFC 9309 says.

    Python's urllib.robotparser takes the *first* rule that matches. zaio.io's
    robots.txt lists "Allow: /" before its Disallow lines, so the standard
    library says /app/ and /course/ are fine to crawl when the site asks for
    them not to be. Here the longest matching rule wins, and Allow wins a tie.
    """

    def __init__(self, text: str, agent: str = "*"):
        self.rules: list[tuple[str, bool]] = []
        groups: dict[str, list[tuple[str, bool]]] = {}
        current: list[str] = []
        in_rules = False
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            key = key.lower()
            if key == "user-agent":
                if in_rules:  # a new group starts
                    current, in_rules = [], False
                current.append(value.lower())
                for name in current:
                    groups.setdefault(name, [])
            elif key in ("allow", "disallow") and current:
                in_rules = True
                if value:  # an empty Disallow means "allow everything"
                    for name in current:
                        groups[name].append((value, key == "allow"))
        self.rules = groups.get(agent.lower(), groups.get("*", []))

    def allowed(self, url: str) -> bool:
        path = urlsplit(url).path or "/"
        best, verdict = -1, True
        for prefix, allow in self.rules:
            if path.startswith(prefix) and (len(prefix) > best or (len(prefix) == best and allow)):
                best, verdict = len(prefix), allow
        return verdict


# --- URLs -------------------------------------------------------------------

def normalize_url(href: str, base: str, site: str = config.WEBSITE_URL) -> str | None:
    """Absolute, canonical URL for a link on the site, or None if it isn't a page to crawl.

    "/tuition-financing/?ref=nav#plans" on any page -> "https://www.zaio.io/tuition-financing".
    Query strings and fragments are dropped: on this site they only select a
    tab or track a click, and keeping them would crawl the same page again.
    """
    href = (href or "").strip()
    if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
        return None
    parts = urlsplit(urljoin(base, href))
    home = urlsplit(site)
    if parts.scheme not in ("http", "https"):
        return None
    if parts.netloc.lower() not in (home.netloc, home.netloc.removeprefix("www.")):
        return None
    path = re.sub(r"/{2,}", "/", parts.path).rstrip("/")
    if FILE_EXTENSIONS.search(path):
        return None
    if any(path == p or path.startswith(p + "/") for p in config.CRAWL_SKIP_PREFIXES):
        return None
    return f"{home.scheme}://{home.netloc}{path}"


def extract_links(html: str, base: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = (normalize_url(a.get("href"), base) for a in soup.find_all("a", href=True))
    return list(dict.fromkeys(link for link in links if link))  # unique, in page order


# --- cleaning ---------------------------------------------------------------

def clean_html(html: str) -> tuple[str, list[Section]]:
    """Rendered HTML -> (page title, sections of main-content text).

    Works on the HTML rather than the browser's innerText on purpose: innerText
    skips anything hidden, and every FAQ answer on zaio.io sits in a collapsed
    accordion. Parsing the HTML keeps those answers.
    """
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    title = re.split(r"\s+[|–—-]\s+Zaio\b", title)[0].strip()  # "Refund Policy | Zaio" -> "Refund Policy"

    for tag in soup.select(", ".join(NOT_TEXT + CHROME)):
        tag.decompose()
    root = soup.find("main") or soup.body or soup
    flatten_grid_tables(soup, root)

    # One line per heading, even when it wraps a <button> (the FAQ questions)
    # or has a <br> in it ("Find the right financing<br/>options for you").
    for tag in root.find_all(HEADINGS):
        tag.string = HEADING_MARK + tag.get_text(" ", strip=True)
    # Every tag boundary is whitespace. The site lays out labels like
    # <span>With</span><span>NQF Level 4</span> with CSS gaps, so without the
    # space they'd read "WithNQF Level 4".
    for tag in root.find_all(True):
        gap = "\n" if tag.name in BLOCKS else " "
        tag.insert_before(gap)
        tag.insert_after(gap)

    sections: list[Section] = []
    heading, lines = "", []

    def close():
        text = clean_text(" ".join(lines))
        if text:
            sections.append(Section(heading=heading, text=text))

    for raw in root.get_text().splitlines():
        line = " ".join(raw.split())
        if line.startswith(HEADING_MARK):
            close()
            heading, lines = line.lstrip(HEADING_MARK).strip(), []
        elif line:
            lines.append(line)
    close()
    return title, sections


def flatten_grid_tables(soup, root) -> None:
    """Rewrite each ARIA grid table as one line per column.

    /compare-courses compares the bootcamps in a CSS grid: column headers,
    then a row header followed by one cell per course. Read top to bottom it
    comes out as "Monthly tuition R4,178 /mo R4,178 /mo R3,213 /mo ...", with
    nothing saying which price is whose. Regrouped per course it reads
    "Data Science: Monthly tuition R3,213 /mo; Upfront tuition R32,950; ...".
    """
    for table in root.select("[role=table]"):
        cells = table.find_all(recursive=False)
        columns = [next(c.stripped_strings, "") for c in cells
                   if c.get("role") == "columnheader" and c.get("aria-hidden") != "true"]
        rows: list[tuple[str, list[str]]] = []
        for cell in cells:
            if cell.get("role") == "rowheader":
                rows.append((cell.get_text(" ", strip=True), []))
            elif cell.get("role") != "columnheader" and rows:
                rows[-1][1].append(cell.get_text(" ", strip=True))
        if not columns or not rows:
            continue

        flat = soup.new_tag("div")
        for i, column in enumerate(columns):
            facts = [f"{label} {values[i]}" for label, values in rows if i < len(values) and values[i]]
            line = soup.new_tag("p")
            line.string = f"{column}: " + "; ".join(facts) + "."
            flat.append(line)
        table.replace_with(flat)


def clean_text(text: str) -> str:
    text = text.replace(HEADING_MARK, "")
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)  # "R 38,950 ." -> "R 38,950."
    return re.sub(r"\s+", " ", text).strip()


def drop_repeated_sections(pages: list[WebPage], min_pages: int = config.REPEATED_SECTION_PAGES) -> list[WebPage]:
    """Keep blocks that appear on many pages only on the first page that has them.

    The course pages share the same financing banner, "join our community"
    block and FAQ. Indexed once per course page, one FAQ answer would fill
    the search results with copies of itself. Pages are in crawl order, so the copy that
    stays is usually the home page's.
    """
    def key(section: Section) -> str:
        return section.text.lower()

    counts: dict[str, int] = {}
    for page in pages:
        for k in {key(s) for s in page.sections}:
            counts[k] = counts.get(k, 0) + 1

    seen: set[str] = set()
    for page in pages:
        kept = []
        for section in page.sections:
            k = key(section)
            if counts[k] >= min_pages and k in seen:
                continue
            seen.add(k)
            kept.append(section)
        page.sections = kept
    return pages


# --- crawling ---------------------------------------------------------------

def crawl(start_url: str = config.WEBSITE_URL, max_pages: int = config.CRAWL_MAX_PAGES,
          delay: float = config.CRAWL_DELAY_SECONDS, log=print) -> list[WebPage]:
    """Breadth-first crawl of the site from the home page, one page at a time."""
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    pages: list[WebPage] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        context = browser.new_context(user_agent=USER_AGENT)
        robots = RobotsRules(context.request.get(urljoin(start_url, "/robots.txt")).text())

        # The sitemap is only used to seed the queue. Pages still have to be
        # reachable by URL and allowed by robots.txt like any other link.
        start = normalize_url(start_url, start_url)
        queue = [start, *sitemap_urls(context, start_url)]
        queued = set(queue)
        visited: set[str] = set()
        tab = context.new_page()

        while queue and len(pages) < max_pages:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)
            if not robots.allowed(url):
                log(f"  skip (robots.txt)  {url}")
                continue
            try:
                response = tab.goto(url, wait_until="domcontentloaded", timeout=30_000)
                try:
                    tab.wait_for_load_state("networkidle", timeout=8_000)
                except PlaywrightError:
                    pass  # analytics can keep the network busy; the content is already there
            except PlaywrightError as exc:
                log(f"  error              {url}  ({exc.message.splitlines()[0]})")
                continue
            if response is None or response.status >= 400:
                log(f"  skip ({response.status if response else 'no response'})  {url}")
                continue

            # Redirects count too: /affiliate-programme lands on /app/getstarted,
            # which robots.txt rules out.
            final = normalize_url(tab.url, tab.url) or url
            if final != url:
                if final in visited:
                    continue
                visited.add(final)
                if not robots.allowed(final):
                    log(f"  skip (redirect to {final}, robots.txt)  {url}")
                    continue

            html = tab.content()
            title, sections = clean_html(html)
            pages.append(WebPage(url=final, title=title, sections=sections))
            log(f"  {len(pages):>3}  {sum(len(s.text) for s in sections):>6} chars  {final}")

            for link in extract_links(html, final):
                if link not in queued:
                    queued.add(link)
                    queue.append(link)
            time.sleep(delay)

        browser.close()
    return pages


def sitemap_urls(context, start_url: str) -> list[str]:
    """Page URLs from /sitemap.xml, following one level of sitemap index."""
    def locs(url: str) -> list[str]:
        try:
            response = context.request.get(url)
        except Exception:
            return []
        if not response.ok:
            return []
        return re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", response.text())

    urls = []
    for loc in locs(urljoin(start_url, "/sitemap.xml")):
        for url in (locs(loc) if loc.endswith(".xml") else [loc]):
            page = normalize_url(url, start_url)
            if page:
                urls.append(page)
    return list(dict.fromkeys(urls))


# --- snapshot file ----------------------------------------------------------

def save_pages(pages: list[WebPage], path: Path = config.WEBSITE_PAGES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "site": config.WEBSITE_URL,
        "crawled_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "pages": [asdict(p) for p in pages],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_pages(path: Path = config.WEBSITE_PAGES_PATH) -> list[WebPage]:
    if not path.exists():
        raise FileNotFoundError(f"No website snapshot at {path}. Run: python -m app.crawl")
    data = json.loads(path.read_text(encoding="utf-8"))
    return [WebPage(url=p["url"], title=p["title"], sections=[Section(**s) for s in p["sections"]])
            for p in data["pages"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl www.zaio.io into data/website/pages.json.")
    parser.add_argument("--max-pages", type=int, default=config.CRAWL_MAX_PAGES)
    parser.add_argument("--out", type=Path, default=config.WEBSITE_PAGES_PATH)
    args = parser.parse_args()

    print(f"Crawling {config.WEBSITE_URL} (up to {args.max_pages} pages)")
    pages = crawl(max_pages=args.max_pages)
    before = sum(len(s.text) for p in pages for s in p.sections)
    pages = [p for p in drop_repeated_sections(pages) if len(p.text) >= config.MIN_PAGE_CHARS]
    after = sum(len(s.text) for p in pages for s in p.sections)
    save_pages(pages, args.out)
    print(f"\nSaved {len(pages)} pages to {args.out}")
    print(f"Text kept after removing repeated blocks: {after:,} of {before:,} characters")


if __name__ == "__main__":
    main()
