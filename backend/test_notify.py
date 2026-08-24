"""Tests for the signup notification.

Nothing here sends an email. What matters is not the sending - that is smtplib's
job - but the three promises around it: that a failure to send cannot lose a
signup, that nothing waits on a mail server, and that a machine with no mail
settings does nothing rather than erroring.

Run from the repo root with:  pytest backend/
"""

import pytest

import notify


@pytest.fixture(autouse=True)
def no_real_mail_settings(monkeypatch):
    """Start every test from an unconfigured machine."""
    for name in ("OWNIT_SMTP_HOST", "OWNIT_SMTP_PORT", "OWNIT_SMTP_USER",
                 "OWNIT_SMTP_PASSWORD", "OWNIT_NOTIFY_TO"):
        monkeypatch.delenv(name, raising=False)


def configured(monkeypatch, **extra):
    monkeypatch.setenv("OWNIT_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("OWNIT_SMTP_USER", "me@example.com")
    monkeypatch.setenv("OWNIT_SMTP_PASSWORD", "an-app-password")
    for name, value in extra.items():
        monkeypatch.setenv(name, value)


# ---- Not being set up is a normal state ------------------------------------


def test_nothing_configured_means_nothing_sent(monkeypatch):
    """Running locally should not mean errors about a mail server that was
    never meant to exist."""
    sent = []
    monkeypatch.setattr(notify, "_send", lambda *a: sent.append(a))

    notify.someone_joined("ada@example.com", "Edexcel A-level", 1)

    assert sent == []


def test_half_configured_counts_as_not_configured(monkeypatch):
    """A host with no password cannot send, and guessing is worse than not."""
    monkeypatch.setenv("OWNIT_SMTP_HOST", "smtp.example.com")

    assert notify._settings() is None


def test_a_full_set_of_settings_is_picked_up(monkeypatch):
    configured(monkeypatch)

    settings = notify._settings()

    assert settings["host"] == "smtp.example.com"
    assert settings["port"] == 587
    # Nowhere to send it was given, so it goes to whoever it came from.
    assert settings["to"] == "me@example.com"


def test_it_can_be_sent_somewhere_else(monkeypatch):
    configured(monkeypatch, OWNIT_NOTIFY_TO="tutor@example.com")

    assert notify._settings()["to"] == "tutor@example.com"


# ---- What it says ----------------------------------------------------------


def test_the_email_says_who_and_what_they_teach(monkeypatch):
    configured(monkeypatch)
    sent = []
    monkeypatch.setattr(notify, "tell", lambda subject, body: sent.append((subject, body)))

    notify.someone_joined("ada@example.com", "Edexcel A-level, Year 13", 7)
    subject, body = sent[0]

    assert "ada@example.com" in subject
    assert "Edexcel A-level, Year 13" in body
    assert "7 on the list" in body


def test_not_saying_what_they_teach_is_said_plainly(monkeypatch):
    """An empty line in an email reads as something having gone wrong."""
    configured(monkeypatch)
    sent = []
    monkeypatch.setattr(notify, "tell", lambda subject, body: sent.append((subject, body)))

    notify.someone_joined("ada@example.com", "   ", 1)

    assert "they did not say" in sent[0][1]


# ---- Not making things worse -----------------------------------------------


def test_a_mail_server_that_refuses_does_not_raise(monkeypatch):
    """The signup is already saved by this point. Losing it because a mail
    server was down would be the worse of the two failures by far."""
    configured(monkeypatch)

    def refuse(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(notify.smtplib, "SMTP", refuse)

    notify._send("subject", "body")   # must simply return


def test_sending_does_not_hold_the_signup_up(monkeypatch):
    """SMTP can take seconds and sometimes hangs. Whoever signed up gets their
    answer immediately either way."""
    configured(monkeypatch)
    started = []
    monkeypatch.setattr(notify.threading, "Thread",
                        lambda **kw: type("T", (), {"start": lambda self: started.append(kw)})())

    notify.tell("subject", "body")

    assert started and started[0]["daemon"] is True


def test_no_thread_is_started_when_there_is_nowhere_to_send(monkeypatch):
    """Otherwise every signup on an unconfigured machine spawns a thread to do
    nothing at all."""
    started = []
    monkeypatch.setattr(notify.threading, "Thread",
                        lambda **kw: type("T", (), {"start": lambda self: started.append(kw)})())

    notify.tell("subject", "body")

    assert started == []


# ---- Being able to tell what happened --------------------------------------


def test_it_says_where_it_would_send(monkeypatch):
    """Startup announces this, so an empty inbox is never a guess about
    whether it was ever switched on."""
    configured(monkeypatch, OWNIT_NOTIFY_TO="tutor@example.com")

    assert notify.sending_to() == "tutor@example.com"


def test_it_says_nothing_when_it_is_not_switched_on(monkeypatch):
    assert notify.sending_to() is None


def test_a_send_that_works_leaves_a_line_in_the_log(monkeypatch, caplog):
    """This was silent. An empty log then meant one of three completely
    different things and there was no way to tell which."""
    import logging
    configured(monkeypatch)

    class Fine:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def starttls(self): pass
        def login(self, *a): pass
        def send_message(self, *a): pass

    monkeypatch.setattr(notify.smtplib, "SMTP", lambda *a, **k: Fine())

    with caplog.at_level(logging.INFO, logger="ownit.notify"):
        notify._send("ownIT: ada@example.com joined the waitlist", "body")

    assert any("sent a notification" in r.message for r in caplog.records)


def test_a_send_that_fails_says_why(monkeypatch, caplog):
    import logging
    configured(monkeypatch)

    def refuse(*a, **k):
        raise OSError("authentication failed")

    monkeypatch.setattr(notify.smtplib, "SMTP", refuse)

    with caplog.at_level(logging.WARNING, logger="ownit.notify"):
        notify._send("subject", "body")

    assert any("authentication failed" in r.message for r in caplog.records)
