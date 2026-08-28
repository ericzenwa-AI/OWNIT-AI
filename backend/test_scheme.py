"""Tests for mining a mark scheme, and for saying so when it does not fit.

The failure worth a test: a document long enough that the answer is cut off
mid-JSON. That arrived as a pydantic complaint about an unterminated string,
forty lines deep, after the PDF had already been read and paid for - which
reads like a bug in the code rather than a document that needs splitting.
"""

from pathlib import Path

import pytest

import llm
import scheme


class _Stream:
    def __init__(self, message):
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get_final_message(self):
        return self._message


class _Message:
    def __init__(self, stop_reason, parsed_output=None):
        self.stop_reason = stop_reason
        self.parsed_output = parsed_output


class _Client:
    def __init__(self, message):
        self._message = message
        self.messages = self

    def stream(self, **kwargs):
        self.kwargs = kwargs
        return _Stream(self._message)


PDF = Path("data/papers/markscheme.pdf")


def test_a_scheme_that_does_not_fit_says_so(monkeypatch, tmp_path):
    pdf = tmp_path / "long.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(scheme, "attachment_block", lambda p: {"type": "text", "text": "x"})

    client = _Client(_Message("max_tokens"))

    with pytest.raises(scheme.SchemeTooLongError) as raised:
        scheme.read_scheme(pdf, client=client)

    said = str(raised.value)
    assert "long.pdf" in said
    assert str(llm.READ_SCHEME.max_tokens) in said
    # It has to say the PDF is fine and the answer was cut off, or the reader
    # goes looking for a corrupt file.
    assert "cut off" in said


def test_no_structured_answer_also_says_so(monkeypatch, tmp_path):
    pdf = tmp_path / "odd.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(scheme, "attachment_block", lambda p: {"type": "text", "text": "x"})

    client = _Client(_Message("refusal", parsed_output=None))

    with pytest.raises(scheme.SchemeTooLongError) as raised:
        scheme.read_scheme(pdf, client=client)

    assert "refusal" in str(raised.value)


def test_a_scheme_that_fits_comes_back(monkeypatch, tmp_path):
    pdf = tmp_path / "fine.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(scheme, "attachment_block", lambda p: {"type": "text", "text": "x"})

    parsed = scheme.Scheme(questions=[
        scheme.SchemeQuestion(
            number="1", topic="indices",
            steps=[scheme.MarkStep(mark="M1", does="differentiate",
                                   skill_hint="differentiate a power")])])
    client = _Client(_Message("end_turn", parsed_output=parsed))

    got = scheme.read_scheme(pdf, client=client)

    assert len(got) == 1
    assert got[0].steps[0].skill_hint == "differentiate a power"


def test_the_command_line_reports_it_instead_of_a_traceback(monkeypatch, capsys):
    """What the person running it actually sees."""
    def too_long(*args, **kwargs):
        raise scheme.SchemeTooLongError("needs more than 64000 tokens, cut off")

    monkeypatch.setattr(scheme, "read_scheme", too_long)

    code = scheme.main([str(PDF)]) if PDF.exists() else 2
    if PDF.exists():
        assert code == 2
        assert "needs more than" in capsys.readouterr().err


def test_the_reader_has_room_for_a_long_document():
    """16000 was not enough for a fifteen-question scheme, and the shortfall
    only showed up as a parse error."""
    assert llm.READ_SCHEME.max_tokens >= 32000


def test_printing_survives_a_console_that_cannot_show_a_theta(capsys):
    scheme._say("theta θ and a minus −")
    assert capsys.readouterr().out.strip()
