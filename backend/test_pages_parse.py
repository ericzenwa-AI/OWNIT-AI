"""Checks the pages the way a browser reads them, not the way a JS engine does.

The maths tests lift the converter out of the page and run it, which says
nothing about whether the browser ever gets that far. It did not: a comment
containing a closing script tag ended the script block early, and two thousand
characters of JavaScript were printed on the page as text. Every other test
passed while the diagnostic was unusable.

The HTML parser ends a script at the first closing tag it sees - inside a
string, inside a comment, it does not matter. So the only way to be sure is to
parse the page as HTML and look at what actually ends up where.
"""

from html.parser import HTMLParser
from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parent.parent / "web"
PAGES = [WEB / "index.html", WEB / "landing.html"]


class Reader(HTMLParser):
    """Collects what a browser would run, and what it would show."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.scripts: list[str] = []
        self.text: list[str] = []
        self._in_script = False
        # Style content is not shown either, and collecting it would fill
        # `visible` with CSS and make the leak check hard to read.
        self._in_style = False

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self._in_script = True
            self.scripts.append("")
        elif tag == "style":
            self._in_style = True

    def handle_endtag(self, tag):
        if tag == "script":
            self._in_script = False
        elif tag == "style":
            self._in_style = False

    def handle_data(self, data):
        if self._in_script:
            self.scripts[-1] += data
        elif not self._in_style:
            self.text.append(data)

    @property
    def visible(self) -> str:
        return " ".join(t.strip() for t in self.text if t.strip())


def read(path: Path) -> Reader:
    reader = Reader()
    reader.feed(path.read_text(encoding="utf-8"))
    return reader


@pytest.mark.parametrize("path", PAGES, ids=lambda p: p.name)
def test_no_javascript_is_shown_as_text(path):
    """The bug this file exists for. Anything that leaked out of a script block
    lands in the body, where a student reads it."""
    visible = read(path).visible

    for giveaway in ("function ", "var ", "return ", "=>", "document.getElementById"):
        assert giveaway not in visible, (
            f"{path.name} is printing JavaScript on the page. "
            f"Most likely a closing script tag inside a string or comment."
        )


@pytest.mark.parametrize("path", PAGES, ids=lambda p: p.name)
def test_the_script_block_survives_to_the_end(path):
    """A script cut short still parses and still passes a JS-engine test, so
    what matters is that the last thing in it is still inside it."""
    scripts = read(path).scripts
    assert scripts, f"{path.name} has no script at all"

    biggest = max(scripts, key=len)
    if path.name == "index.html":
        assert "askScreen()" in biggest
        assert "function mathHtml" in biggest
    else:
        assert "api/waitlist" in biggest
        assert "api/comment" in biggest


@pytest.mark.parametrize("path", PAGES, ids=lambda p: p.name)
def test_no_stray_closing_script_tag(path):
    """Said directly, so the reason is obvious when it fails: the parser ends a
    script at the first closing tag it meets, comment or not."""
    source = path.read_text(encoding="utf-8")

    assert source.count("</script>") == source.count("<script"), (
        f"{path.name} has a closing script tag with no opening one. Inside JS "
        f"it must be written some other way - split, escaped, or described."
    )
