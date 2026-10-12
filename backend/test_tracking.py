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
    assert all(set(row) == {"day", "loads", "people", "started"} for row in days)


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
    assert "Who is arriving" in response.text
    assert "Where they came from" in response.text
    assert "What people pressed" in response.text


def test_the_dashboard_draws_the_days_once_there_are_some(client, db, monkeypatch):
    monkeypatch.setenv("OWNIT_ADMIN_PASSWORD", "secret")
    client.get("/", headers={"user-agent": "x"})

    text = client.get("/admin/numbers", auth=("ownit", "secret")).text

    assert "<svg" in text and "People a day" in text
