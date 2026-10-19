"""Counting visits, where they came from, and what got pressed.

The funnel could already see pages and sessions. It could not see the space
between them: somebody who read the whole front page and never pressed the
button looked exactly like somebody who shut it at once. It also could not see
where anybody came from, which makes a round of emails unmeasurable.

The thing to be careful about is that this counts people without knowing who
they are. The hash exists to say "these four loads were one person this
morning" and must be unable to say anything else - no cookie, no stored
address, and nothing that survives the day.
"""

import pytest
from fastapi.testclient import TestClient

import api
import store


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DEFAULT_PATH", tmp_path / "tracking.db")
    connection = store.connect()
    yield connection
    connection.close()


@pytest.fixture
def client(db):
    return TestClient(api.app)


# ---- what it must never do -------------------------------------------------


def test_no_cookie_is_ever_set(client):
    """The whole point of the rotating hash is not needing one."""
    for path in ("/", "/student", "/start"):
        assert not client.get(path).cookies, f"{path} set a cookie"


def test_the_address_is_not_stored(client, db):
    client.get("/", headers={"x-forwarded-for": "203.0.113.9", "user-agent": "a"})

    row = db.execute("SELECT * FROM page_view ORDER BY id DESC LIMIT 1").fetchone()
    assert "203.0.113.9" not in str(dict(row)), "the address itself was kept"
    assert row["visitor"], "but something has to tie one person's loads together"


def test_the_hash_does_not_survive_the_day(db):
    """Yesterday's salt is gone, so yesterday's hashes cannot be recomputed or
    matched against today's."""
    today = store.salt_for_today(db)
    db.execute("INSERT OR REPLACE INTO visitor_salt (day, value) VALUES ('2020-01-01', 'old')")
    db.commit()

    store.forget_old_salts(db, keep_days=3)

    left = [row["day"] for row in db.execute("SELECT day FROM visitor_salt")]
    assert "2020-01-01" not in left
    assert store.salt_for_today(db) == today, "today's must not be rotated out"


def test_a_made_up_event_name_is_refused(client, db):
    """The endpoint is open to anybody, so the name is an allowlist."""
    response = client.post("/api/event", json={"name": "drop table", "page": "/"})

    assert response.status_code == 200, "and refused politely, not with an error"
    assert db.execute("SELECT COUNT(*) n FROM event").fetchone()["n"] == 0


def test_counting_failing_never_breaks_a_page(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("the counter is down")

    monkeypatch.setattr(store, "record_page_view", boom)
    assert client.get("/").status_code == 200


def test_counting_failing_never_breaks_an_event(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("the counter is down")

    monkeypatch.setattr(store, "record_event", boom)
    assert client.post("/api/event", json={"name": "cta_start"}).status_code == 200


# ---- what it records -------------------------------------------------------


def test_one_browser_in_one_day_is_one_person(client, db):
    headers = {"x-forwarded-for": "198.51.100.4", "user-agent": "Firefox"}
    for _ in range(3):
        client.get("/", headers=headers)
    client.get("/", headers={"x-forwarded-for": "198.51.100.5", "user-agent": "Firefox"})

    seen = store.visitors(db)
    assert seen["loads"] == 4
    assert seen["people"] == 2


def test_the_tag_on_the_link_is_what_gets_credited(client, db):
    client.get("/?utm_source=email", headers={"user-agent": "x"})

    assert db.execute(
        "SELECT source FROM page_view ORDER BY id DESC LIMIT 1").fetchone()["source"] == "email"


def test_a_referring_site_is_reduced_to_its_host(client, db):
    """A full referring URL can carry a search query or a private page title,
    and none of that is any use for counting."""
    client.get("/", headers={"referer": "https://www.mumsnet.com/talk/1234?q=secret",
                             "user-agent": "x"})

    assert db.execute(
        "SELECT source FROM page_view ORDER BY id DESC LIMIT 1").fetchone()["source"] == "mumsnet.com"


def test_our_own_pages_are_not_traffic_from_somewhere(client, db):
    client.get("/start", headers={"referer": "https://ownitmaths.com/", "user-agent": "x"})

    assert db.execute(
        "SELECT source FROM page_view ORDER BY id DESC LIMIT 1").fetchone()["source"] == "internal"


def test_no_tag_and_no_referrer_is_direct(client, db):
    client.get("/", headers={"user-agent": "x"})

    assert db.execute(
        "SELECT source FROM page_view ORDER BY id DESC LIMIT 1").fetchone()["source"] == "direct"


def test_a_press_is_recorded_with_where_it_came_from(client, db):
    client.post("/api/event", json={"name": "cta_start", "page": "/", "detail": "Try it"},
                headers={"utm": "x", "referer": "https://news.ycombinator.com/x",
                         "user-agent": "x"})

    row = db.execute("SELECT * FROM event").fetchone()
    assert row["name"] == "cta_start"
    assert row["source"] == "news.ycombinator.com"
    assert row["detail"] == "Try it"


def test_clicks_add_up(client, db):
    for _ in range(2):
        client.post("/api/event", json={"name": "cta_start"}, headers={"user-agent": "x"})
    client.post("/api/event", json={"name": "cta_how"}, headers={"user-agent": "x"})

    counted = {row["name"]: row["times"] for row in store.clicks(db)}
    assert counted == {"cta_start": 2, "cta_how": 1}


def test_every_day_in_the_range_is_there_even_when_empty(db):
    """A gap that silently closes up reads as a quiet day rather than no day."""
    days = store.by_day(db, days=14)

    assert len(days) == 14
    assert days[0]["day"] < days[-1]["day"], "oldest first"
    assert all(set(row) == {"day", "loads", "people", "started", "finished"}
               for row in days), "finished was added for the dashboard"


def test_the_student_page_is_counted_separately(client, db):
    client.get("/student", headers={"user-agent": "x"})
    client.get("/", headers={"user-agent": "x"})

    figures = store.how_it_is_going(db)
    assert figures["student"] == 1
    assert figures["landing"] == 1


# ---- the pages and the dashboard -------------------------------------------


def _page(name):
    import pathlib

    return (pathlib.Path(__file__).resolve().parent.parent
            / "web" / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["landing.html", "student.html", "index.html"])
def test_every_page_can_report(name):
    assert "function track(" in _page(name)


def test_the_buttons_that_matter_are_tagged():
    landing = _page("landing.html")
    assert landing.count('data-track="cta_start"') >= 3, "every way in"
    assert 'data-track="cta_student"' in landing
    assert 'data-track="cta_tutor"' in landing
    assert 'data-track="cta_start"' in _page("student.html")


def test_the_app_reports_the_moments_that_matter():
    app = _page("index.html")
    for moment in ("sent_question", "report_seen", "step_opened", "retake",
                   "another_question"):
        assert f"track('{moment}'" in app, f"{moment} is not counted"


def test_every_tracked_name_is_one_the_server_accepts():
    """A page reporting a name the server drops is a number that silently
    never appears."""
    import re

    used = set()
    for name in ("landing.html", "student.html", "index.html"):
        page = _page(name)
        used |= set(re.findall(r'data-track="([a-z_]+)"', page))
        used |= set(re.findall(r"track\('([a-z_]+)'", page))

    unknown = sorted(used - set(store.EVENTS))
    assert not unknown, f"the server would drop: {unknown}"


def test_the_dashboard_renders_with_nothing_in_it(client, db, monkeypatch):
    """The first time it is opened there is no data at all, which is exactly
    when a dashboard built around having some falls over."""
    monkeypatch.setenv("OWNIT_ADMIN_PASSWORD", "secret")

    response = client.get("/admin/numbers", auth=("ownit", "secret"))

    assert response.status_code == 200
    assert "From arriving to being taught" in response.text
    assert "Where they came from" in response.text
    assert "What they pressed" in response.text


def test_the_dashboard_draws_the_days_once_there_are_some(client, db, monkeypatch):
    monkeypatch.setenv("OWNIT_ADMIN_PASSWORD", "secret")
    client.get("/", headers={"user-agent": "x"})

    text = client.get("/admin/numbers", auth=("ownit", "secret")).text

    assert "<svg" in text and "People a day" in text


# ---- the dashboard ---------------------------------------------------------


def _numbers(client, monkeypatch, **params):
    monkeypatch.setenv("OWNIT_ADMIN_PASSWORD", "secret")
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return client.get(f"/admin/numbers{'?' + query if query else ''}",
                      auth=("admin", "secret"))


def test_the_dashboard_needs_the_password(client, monkeypatch):
    monkeypatch.setenv("OWNIT_ADMIN_PASSWORD", "secret")

    assert client.get("/admin/numbers").status_code == 401
    assert client.get("/admin/numbers", auth=("a", "wrong")).status_code == 401


def test_it_renders_on_a_database_with_nothing_in_it(client, db, monkeypatch):
    """Which is exactly when a dashboard built around having data falls over."""
    page = _numbers(client, monkeypatch)

    assert page.status_code == 200
    for panel in ("From arriving to being taught", "People a day",
                  "Where they came from", "What they pressed",
                  "What it keeps finding"):
        assert panel in page.text


def test_the_number_that_matters_is_the_headline(client, db, monkeypatch):
    """How many of the people who arrive actually reach a finding."""
    assert "reached a finding" in _numbers(client, monkeypatch).text


def test_each_funnel_step_is_shown_against_the_one_above():
    import dashboard

    drawn = dashboard.funnel([("Top", 10, ""), ("Next", 5, ""), ("Last", 1, "")])

    assert "50% of the step above" in drawn
    assert "20% of the step above" in drawn


def test_a_share_of_nothing_is_not_printed_as_a_percentage():
    """A row showing zero with "20% of all" under it reads as a
    contradiction, and that is what the old fallback produced."""
    import dashboard

    drawn = dashboard.funnel([("Top", 0, ""), ("Next", 5, "")])

    assert "nobody reached the step above" in drawn
    assert "of the step above" not in drawn.replace(
        "nobody reached the step above", ""), "no percentage of nothing"


def test_more_than_the_step_above_is_said_rather_than_shown_as_140_percent():
    import dashboard

    drawn = dashboard.funnel([("Top", 10, ""), ("Next", 14, "")])

    assert "more than the step above" in drawn


def test_a_bar_with_a_real_number_is_never_invisible():
    """One person and nobody have to look different."""
    import dashboard

    drawn = dashboard.bars([("big", 500, ""), ("one", 1, "")])

    assert "width:0%" not in drawn and "width:0.0%" not in drawn


def test_bars_are_block_elements_so_they_actually_draw():
    """They were spans, which are inline - the height and the width did
    nothing and every bar rendered as an empty track."""
    import dashboard

    assert ".bar .fill { display: block;" in dashboard.STYLE
    assert ".bar .track { display: block;" in dashboard.STYLE


def test_the_time_filter_works(client, db, monkeypatch):
    for days in (7, 30):
        assert _numbers(client, monkeypatch, days=days).status_code == 200
    assert "7 days" in _numbers(client, monkeypatch).text


def test_the_dashboard_shows_no_personal_data(client, db, monkeypatch):
    """It is counts. A visitor hash or an address appearing on it would mean
    the page had started holding something about somebody."""
    client.get("/", headers={"user-agent": "Firefox", "x-forwarded-for": "203.0.113.7"})

    page = _numbers(client, monkeypatch).text

    assert "203.0.113.7" not in page
    assert "Firefox" not in page


# ---- crawlers are not people -----------------------------------------------


@pytest.mark.parametrize("agent", [
    "curl/8.4.0",
    "Googlebot/2.1 (+http://www.google.com/bot.html)",
    "Mozilla/5.0 (compatible; AhrefsBot/7.0; +http://ahrefs.com/robot/)",
    "python-requests/2.31.0",
    "Mozilla/5.0 (X11; Linux x86_64) HeadlessChrome/120",
    "",
])
def test_a_crawler_is_not_counted_as_a_visit(client, db, agent):
    """The site is linked from a directory, so it is indexed, scraped and
    monitored around the clock, and every deploy check from a terminal
    counted as somebody new. The browser string is matched and thrown away -
    it is never stored - so this has to happen at the moment of counting."""
    # Explicitly, even when empty: left out, the test client supplies its
    # own browser string and the empty case would never be exercised.
    client.get("/", headers={"user-agent": agent})

    assert store.visitors(db)["loads"] == 0


def test_a_real_browser_still_counts(client, db):
    client.get("/", headers={
        "user-agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                      "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
                      "Mobile/15E148 Safari/604.1"})

    assert store.visitors(db)["people"] == 1


def test_turning_a_crawler_away_still_serves_the_page(client):
    """Counting is never allowed to decide whether somebody sees the site."""
    assert client.get("/", headers={"user-agent": "Googlebot/2.1"}).status_code == 200
