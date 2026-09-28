from urllib.robotparser import RobotFileParser

import pytest

from app import config
from app.crawl import (RobotsRules, Section, WebPage, clean_html, drop_repeated_sections,
                       extract_links, load_pages, normalize_url, save_pages)

SITE = "https://www.zaio.io"

# Trimmed from https://www.zaio.io/robots.txt
ROBOTS = """\
# The trailing slashes are load-bearing: /course/ must not catch /course-instructors,
# and /app/ must not catch /apply-now.
User-agent: *
Allow: /
Disallow: /app/
Disallow: /static/
Disallow: /course/
Disallow: /welcome
Disallow: /home

User-agent: GPTBot
Allow: /
Disallow: /app/

Sitemap: https://www.zaio.io/sitemap.xml
"""

# Shaped like the real pages: <header>/<main>/<footer>, a heading with a <br>,
# labels built from sibling <span>s, and a Bootstrap FAQ accordion whose
# answers are in the DOM but collapsed.
PAGE = """\
<html><head><title>Refund Policy | Zaio</title><script>window.__NEXT_DATA__ = {}</script>
<style>.x{color:red}</style></head>
<body>
<header class="Header_menuArea"><a href="/bootcamps">Bootcamps</a><a href="/aboutus">About us</a></header>
<main>
  <h1>Find the right financing<br/>options for you</h1>
  <p>Pay <strong>R 38,950</strong> upfront .</p>
  <div class="meta"><span>With</span><span>NQF Level 4 certification</span></div>
  <svg><text>arrow-icon</text></svg>
  <h2>Frequently Asked Questions</h2>
  <div class="accordion-item">
    <h2 class="accordion-header"><button aria-expanded="false">How do the live classes work?</button></h2>
    <div class="accordion-collapse collapse"><div class="accordion-body">Twice a week, once in the
      morning and once in the evening.</div></div>
  </div>
</main>
<div class="CookieBanner">We use cookies. Accept?</div>
<footer class="Footer_footer"><a href="/terms">Terms</a> <p>Copyright 2026 Zaio</p></footer>
</body></html>
"""


# --- robots.txt -------------------------------------------------------------

@pytest.fixture
def robots():
    return RobotsRules(ROBOTS)


@pytest.mark.parametrize("path", ["/", "/bootcamps", "/apply-now", "/app-next-steps", "/course-instructors"])
def test_robots_allows_normal_pages(robots, path):
    assert robots.allowed(SITE + path)


@pytest.mark.parametrize("path", ["/app/", "/app/dashboard", "/course/123", "/static/logo.png", "/home", "/welcome"])
def test_robots_blocks_disallowed_paths(robots, path):
    assert not robots.allowed(SITE + path)


def test_standard_library_gets_this_file_wrong():
    # Why RobotsRules exists: urllib stops at the first matching rule, and
    # "Allow: /" comes first, so it would crawl /app/.
    stdlib = RobotFileParser()
    stdlib.parse(ROBOTS.splitlines())
    assert stdlib.can_fetch("*", SITE + "/app/dashboard")
    assert not RobotsRules(ROBOTS).allowed(SITE + "/app/dashboard")


def test_robots_uses_the_group_for_a_named_agent():
    gptbot = RobotsRules(ROBOTS, agent="GPTBot")
    assert not gptbot.allowed(SITE + "/app/x")
    assert gptbot.allowed(SITE + "/course/123")  # only /app/ is disallowed for GPTBot


def test_allow_wins_a_tie_and_longest_rule_wins():
    rules = RobotsRules("User-agent: *\nDisallow: /docs\nAllow: /docs\nDisallow: /a\nAllow: /a/public")
    assert rules.allowed(SITE + "/docs/x")
    assert rules.allowed(SITE + "/a/public/page")
    assert not rules.allowed(SITE + "/a/private")


def test_empty_robots_allows_everything():
    assert RobotsRules("").allowed(SITE + "/anything")


# --- URLs -------------------------------------------------------------------

@pytest.mark.parametrize("href, base, expected", [
    ("/tuition-financing/?ref=nav#plans", SITE + "/bootcamps", SITE + "/tuition-financing"),
    ("https://zaio.io/aboutus", SITE, SITE + "/aboutus"),
    ("https://www.zaio.io/", SITE, SITE),
    ("data-science", SITE + "/tuition/", SITE + "/tuition/data-science"),
    ("//www.zaio.io//refundpolicy", SITE, SITE + "/refundpolicy"),
    ("/blogger-programme", SITE, SITE + "/blogger-programme"),  # only /blog itself is skipped
])
def test_normalize_url(href, base, expected):
    assert normalize_url(href, base) == expected


@pytest.mark.parametrize("href", [
    "https://www.linkedin.com/company/zaio", "mailto:hello@zaio.io", "tel:+27210000000",
    "javascript:void(0)", "#faq", "", "/brochure.pdf", "/images/hero.webp",
    "/blog", "/blog/best-coding-bootcamps-south-africa-2025/",
])
def test_links_that_are_not_pages_are_dropped(href):
    assert normalize_url(href, SITE) is None


def test_extract_links_is_unique_and_in_page_order():
    html = '<a href="/b">B</a><a href="/a">A</a><a href="/b#x">B again</a><a href="https://x.com">X</a>'
    assert extract_links(html, SITE) == [SITE + "/b", SITE + "/a"]


# --- cleaning ---------------------------------------------------------------

def test_clean_html_keeps_main_content_only():
    title, sections = clean_html(PAGE)
    text = " ".join(s.text for s in sections)

    assert title == "Refund Policy"
    for chrome in ["Bootcamps", "About us", "Terms", "Copyright", "cookies", "__NEXT_DATA__", "color:red", "arrow-icon"]:
        assert chrome not in text


def test_clean_html_splits_on_headings():
    _, sections = clean_html(PAGE)
    assert [s.heading for s in sections] == ["Find the right financing options for you",
                                             "How do the live classes work?"]


def test_collapsed_faq_answers_are_kept():
    # innerText would drop these: the accordion body is hidden until clicked.
    _, sections = clean_html(PAGE)
    assert sections[1].text == "Twice a week, once in the morning and once in the evening."


def test_inline_tags_keep_their_spacing():
    _, sections = clean_html(PAGE)
    assert sections[0].text == "Pay R 38,950 upfront. With NQF Level 4 certification"


def test_heading_without_text_below_it_is_dropped():
    _, sections = clean_html(PAGE)
    assert "Frequently Asked Questions" not in [s.heading for s in sections]


def test_grid_table_is_regrouped_per_column():
    # /compare-courses: a CSS grid of column headers, then a row header
    # followed by one cell per course.
    html = """<main><h2>Compare bootcamps</h2>
    <div role="table">
      <div role="columnheader" aria-hidden="true"></div>
      <div role="columnheader"><div>Fullstack AI Engineer</div><span>Most popular</span></div>
      <div role="columnheader"><div>Data Science</div></div>
      <div role="rowheader">Monthly tuition</div>
      <div><span>R4,178</span><span>/mo</span></div><div><span>R3,213</span><span>/mo</span></div>
      <div role="rowheader">Duration</div>
      <div>±24 weeks</div><div>±30 weeks</div>
    </div></main>"""
    _, [section] = clean_html(html)
    assert section.text == ("Fullstack AI Engineer: Monthly tuition R4,178 /mo; Duration ±24 weeks. "
                            "Data Science: Monthly tuition R3,213 /mo; Duration ±30 weeks.")


def test_page_without_main_falls_back_to_body():
    _, sections = clean_html("<html><body><header>Menu</header><h1>Terms</h1><p>Be nice.</p></body></html>")
    assert [(s.heading, s.text) for s in sections] == [("Terms", "Be nice.")]


def test_repeated_blocks_are_kept_once():
    banner = Section("Join our community", "Unlock the benefits of our Discord community.")
    faq = Section("Can I study while working?", "Yes, classes are recorded.")
    pages = [WebPage(f"{SITE}/p{i}", f"Page {i}", [Section(f"Page {i}", f"Unique text {i}."), banner]) for i in range(4)]
    pages[0].sections.append(faq)
    pages[1].sections.append(faq)  # on two pages only: under the threshold of 3

    pages = drop_repeated_sections(pages, min_pages=3)

    assert [banner in p.sections for p in pages] == [True, False, False, False]
    assert [faq in p.sections for p in pages] == [True, True, False, False]
    assert all(p.sections[0].text.startswith("Unique text") for p in pages)


def test_snapshot_round_trip(tmp_path):
    pages = [WebPage(SITE + "/refundpolicy", "Refund Policy", [Section("Refunds", "Within 14 days.")])]
    path = tmp_path / "pages.json"
    save_pages(pages, path)
    assert load_pages(path) == pages


def test_missing_snapshot_says_how_to_make_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m app.crawl"):
        load_pages(tmp_path / "nope.json")


# --- the committed snapshot -------------------------------------------------

@pytest.fixture(scope="module")
def snapshot():
    return load_pages(config.WEBSITE_PAGES_PATH)


def test_snapshot_only_has_allowed_pages_of_the_main_site(snapshot):
    robots = RobotsRules(ROBOTS)
    urls = [p.url for p in snapshot]

    assert len(urls) == len(set(urls)) >= 30
    assert all(url == SITE or url.startswith(SITE + "/") for url in urls)
    assert not [u for u in urls if u.startswith(SITE + "/blog")]
    assert all(robots.allowed(u) for u in urls)


def test_snapshot_has_no_empty_pages(snapshot):
    assert all(len(p.text) >= config.MIN_PAGE_CHARS for p in snapshot)


def test_snapshot_kept_the_home_page_faq(snapshot):
    home = next(p for p in snapshot if p.url == SITE)
    faq = next(s for s in home.sections if s.heading == "How do the live classes work?")
    assert "twice a week" in faq.text
