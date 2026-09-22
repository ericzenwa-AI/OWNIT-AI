"""A backtick in a comment inside a template takes the whole page down.

The page's screens are JavaScript template literals, and the HTML comments
explaining them live inside those literals. A backtick in one of those comments
- `plain`, written the way this codebase names a field everywhere else - ends
the template early. The rest of the script is then a syntax error, nothing on
the page runs, and every student opening /start gets a blank screen.

That shipped on 2026-09-19 and stayed live for three days, because every other
test here reads the page as text and a page that is text-correct and
script-broken passes all of them.
"""

import pathlib
import re

WEB = pathlib.Path(__file__).resolve().parent.parent / "web"


def _script(name):
    text = (WEB / name).read_text(encoding="utf-8")
    return "\n".join(re.findall(r"<script>(.*?)</script>", text, re.S))


def test_no_html_comment_inside_the_script_contains_a_backtick():
    script = _script("index.html")
    bad = [c for c in re.findall(r"<!--(.*?)-->", script, re.S) if "`" in c]
    assert not bad, (
        "a backtick inside an HTML comment in a template ends the template and "
        "blanks the page. Use quotes instead:\n"
        + "\n---\n".join(c.strip()[:160] for c in bad))


def test_the_script_has_an_even_number_of_backticks():
    """A cruder net for the same fault anywhere else in the script: every
    template that opens has to close. Escaped backticks are not delimiters."""
    script = re.sub(r"\\`", "", _script("index.html"))
    # Line comments may mention a backtick in prose; they are not templates.
    script = re.sub(r"^\s*//.*$", "", script, flags=re.M)
    assert script.count("`") % 2 == 0, "an unclosed template literal"
