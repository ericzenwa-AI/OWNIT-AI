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

    OWNIT_SMTP_HOST      smtp.gmail.com
    OWNIT_SMTP_PORT      587
    OWNIT_SMTP_USER      the address it sends from
    OWNIT_SMTP_PASSWORD  an app password, not the account password
    OWNIT_NOTIFY_TO      where to send it (defaults to the user above)
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

    return {
        "host": host,
        "port": int(os.environ.get("OWNIT_SMTP_PORT", "587")),
        "user": user,
        "password": password,
        "to": os.environ.get("OWNIT_NOTIFY_TO", "").strip() or user,
    }


def _send(subject: str, body: str) -> None:
    """Actually send it. Runs on its own thread; failures are logged only."""
    settings = _settings()
    if settings is None:
        return

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = settings["user"]
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
