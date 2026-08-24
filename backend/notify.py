"""Sends a plain email when something happens worth knowing about.

Built so nobody has to remember to open a page. A signup that sits unseen for a
week is a tutor who asked to hear from you and did not, which is worse than not
having asked them.

Three rules, all of them about not making things worse:

  It never breaks what it is reporting on. A signup is saved before this is
  called, and if the send fails the person is still on the list - they simply
  do not know that you did not find out.

  It never holds anything up. SMTP can take several seconds and sometimes
  hangs; it runs on its own thread so the person who signed up gets their
  answer immediately either way.

  With nothing configured it does nothing, quietly. Running locally, or in
  tests, should not mean errors about a mail server that was never meant to
  exist.

Configure with, in the Render dashboard:

    OWNIT_SMTP_HOST      smtp.gmail.com, or smtp.resend.com
    OWNIT_SMTP_PORT      587
    OWNIT_SMTP_USER      what the server wants as a username. With Gmail that
                         is your address; with Resend it is the word "resend"
    OWNIT_SMTP_PASSWORD  an app password, or an API key
    OWNIT_SMTP_FROM      the address it is sent from. Only needed when that is
                         not the username - onboarding@resend.dev, say
    OWNIT_NOTIFY_TO      where to send it (defaults to the from address)
"""

from __future__ import annotations

import logging
import os
import smtplib
import threading
from email.message import EmailMessage

log = logging.getLogger("ownit.notify")

TIMEOUT_SECONDS = 20


def _settings() -> dict | None:
    """What is needed to send, or None if this is not set up."""
    host = os.environ.get("OWNIT_SMTP_HOST", "").strip()
    user = os.environ.get("OWNIT_SMTP_USER", "").strip()
    password = os.environ.get("OWNIT_SMTP_PASSWORD", "").strip()
    if not (host and user and password):
        return None

    # Who it is from is not always who is logging in. With Gmail the username
    # is the address, so defaulting to it works; with Resend the username is
    # the literal word "resend" and the API key is the password, so sending
    # from it produced "Bad sender address syntax" and nothing arrived.
    sender = os.environ.get("OWNIT_SMTP_FROM", "").strip()
    if "@" not in sender:
        sender = user if "@" in user else ""

    if not sender:
        return None

    return {
        "host": host,
        "port": int(os.environ.get("OWNIT_SMTP_PORT", "587")),
        "user": user,
        "password": password,
        "from": sender,
        "to": os.environ.get("OWNIT_NOTIFY_TO", "").strip() or sender,
    }


def sending_to() -> str | None:
    """Where notifications go, or None if this is not switched on.

    Exists so startup can say which it is. An empty log should not be the only
    evidence that nothing was ever going to be sent.
    """
    settings = _settings()
    if not settings:
        return None
    return f"{settings['to']} (from {settings['from']})"


def _send(subject: str, body: str) -> None:
    """Actually send it. Runs on its own thread; failures are logged only."""
    settings = _settings()
    if settings is None:
        return

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = settings["from"]
    message["To"] = settings["to"]
    message.set_content(body)

    try:
        with smtplib.SMTP(settings["host"], settings["port"],
                          timeout=TIMEOUT_SECONDS) as server:
            server.starttls()
            server.login(settings["user"], settings["password"])
            server.send_message(message)
        log.info("sent a notification: %s", subject)
    except Exception as error:  # noqa: BLE001 - a mail server is not our problem
        log.warning("could not send a notification (%s): %s", subject, error)


def tell(subject: str, body: str) -> None:
    """Send in the background, or do nothing if this is not configured.

    Checked before the thread is started rather than inside it, so an
    unconfigured machine does not spawn a thread per signup to do nothing.
    """
    if _settings() is None:
        return
    threading.Thread(target=_send, args=(subject, body), daemon=True).start()


def check() -> dict:
    """Try to send one email now, and say exactly what happened.

    Everything else about sending is deliberately quiet and out of the way: it
    runs on another thread, it swallows its own failures, and it must never
    hold up a signup. All of which makes it impossible to answer "did that
    work?" without reading logs and guessing.

    This does the opposite. It sends on this thread, waits, and reports. The
    password is never in what comes back.
    """
    settings = _settings()
    if settings is None:
        return {
            "configured": False,
            "sent": False,
            "why": ("No usable mail settings. OWNIT_SMTP_HOST, OWNIT_SMTP_USER "
                    "and OWNIT_SMTP_PASSWORD must all be set, and the sender "
                    "must be an address - set OWNIT_SMTP_FROM when the username "
                    "is not one, as with Resend."),
        }

    where = {
        "host": settings["host"],
        "port": settings["port"],
        "username": settings["user"],
        "from": settings["from"],
        "to": settings["to"],
    }

    message = EmailMessage()
    message["Subject"] = "ownIT: test"
    message["From"] = settings["from"]
    message["To"] = settings["to"]
    message.set_content(
        "This is the test from /admin/email.\n\n"
        "If it arrived, waitlist signups will too.\n"
    )

    try:
        with smtplib.SMTP(settings["host"], settings["port"],
                          timeout=TIMEOUT_SECONDS) as server:
            server.starttls()
            server.login(settings["user"], settings["password"])
            server.send_message(message)
    except Exception as error:  # noqa: BLE001 - the reason is the whole point
        log.warning("test email failed: %s", error)
        return {"configured": True, "sent": False, "why": f"{type(error).__name__}: {error}", **where}

    log.info("test email sent to %s", settings["to"])
    return {"configured": True, "sent": True, "why": None, **where}


def someone_joined(email: str, studying: str | None, total: int) -> None:
    """A new name on the waitlist."""
    teaches = (studying or "").strip() or "(they did not say)"
    tell(
        subject=f"ownIT: {email} joined the waitlist",
        body=(
            f"{email} just asked to be told when ownIT opens.\n\n"
            f"What they teach: {teaches}\n\n"
            f"That makes {total} on the list.\n"
        ),
    )
