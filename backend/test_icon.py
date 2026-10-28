"""The tab icon: that it is served, and that it does not poison the numbers.

Two things made this worth pinning rather than eyeballing once.

The first is that the icon is three files which must stay one mark. The SVG
is hand-written and the two bitmaps are drawn by backend/icons.py from numbers
copied into it, so the day somebody nudges the SVG and forgets the script, the
tab and the home screen quietly disagree. The test reads both and compares.

The second is the funnel. A browser fetches /favicon.ico on its own, for a
page nobody opened and sometimes for no page at all, so if an icon fetch were
counted as a visit then every crawler, every link preview and every tab
restore would show up as a person. That is the kind of error that does not
look like an error - the numbers just get better - so it gets a test.
"""

import re

import pytest
from fastapi.testclient import TestClient

import api
import icons
import store

ICON_PATHS = ("/icon.svg", "/favicon.ico", "/apple-touch-icon.png")


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DEFAULT_PATH", tmp_path / "icon.db")
    connection = store.connect()
    yield connection
    connection.close()


@pytest.fixture
def client(db):
    return TestClient(api.app)


# ---- it is there -----------------------------------------------------------


@pytest.mark.parametrize("path", ICON_PATHS)
def test_each_icon_is_served(client, path):
    assert client.get(path).status_code == 200, f"{path} is not being served"


def test_the_ico_holds_the_sizes_a_browser_asks_for(client):
    """A single-size .ico gets scaled by the browser and looks it."""
    from io import BytesIO

    from PIL import Image

    held = {size for size, _ in Image.open(BytesIO(
        client.get("/favicon.ico").content)).info["sizes"]}
    for size in icons.ICO_SIZES:
        assert size in held, f"the .ico has no {size}px frame"


@pytest.mark.parametrize("page", ("/", "/student", "/start"))
def test_every_page_asks_for_the_icon(client, page):
    """A page without the link tags shows a blank sheet in the tab."""
    html = client.get(page).text
    assert 'href="/icon.svg"' in html, f"{page} does not point at the icon"
    assert 'href="/favicon.ico"' in html, f"{page} has no bitmap fallback"
    assert "apple-touch-icon" in html, f"{page} has no home screen icon"


# ---- it is one mark --------------------------------------------------------


def test_the_bitmaps_are_drawn_from_the_svg_geometry():
    """If the two drift, the tab and the home screen stop matching."""
    svg = (api.WEB / "icon.svg").read_text(encoding="utf-8")
    bars = [tuple(float(n) for n in row) for row in re.findall(
        r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"',
        svg)]
    assert bars == [tuple(bar) for bar in icons.BARS], (
        "web/icon.svg and backend/icons.py have drifted - "
        "change one and re-run `python backend/icons.py`")

    lit = f"#{''.join(f'{part:02X}' for part in icons.LIT)}"
    assert lit.lower() in svg.lower(), "the lit bar is a different colour"


# ---- it does not become a visitor ------------------------------------------


@pytest.mark.parametrize("path", ICON_PATHS)
def test_fetching_an_icon_is_not_a_visit(client, db, path):
    before = store.visitors(db)
    client.get(path)
    assert store.visitors(db) == before, (
        f"{path} was counted as somebody looking at the site")


def test_an_icon_may_be_cached_but_a_page_may_not(client):
    """Opposite needs: a stale icon is cosmetic, a stale app is a bug."""
    assert "max-age" in client.get("/icon.svg").headers["cache-control"]
    assert "no-cache" in client.get("/").headers["cache-control"]
