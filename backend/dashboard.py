"""The numbers, drawn.

The old page was a stack of monospace tables. Everything was in it and
nothing stood out, which for the one number that matters - how many people
who arrive actually reach a finding - is the same as not having it.

HOW IT IS BUILT

Hand-written SVG, no chart library. Three reasons: this page sits behind a
password on a server with no build step, a library would be a third-party
request on a page showing private numbers, and every chart here is bars -
which is forty lines of arithmetic, not a dependency.

WHAT IS DRAWN AND WHY

A bar per thing, laid out horizontally where the labels are words, because a
rotated axis label is a chart apologising for its own layout. One series per
chart: two measures on one axis invite reading one against the other, and the
pairs here (people and diagnoses, for instance) are not comparable. Where two
belong together they are two small charts side by side instead.

COLOUR

Violet is the ordinary measure. Blue is the second one in a pair. Green and
rust mean what they mean everywhere else in this product - a thing that held,
a thing that broke - and are never spent on decoration. That is why the
funnel is one hue going down and not a rainbow: the steps are the same kind
of thing, so they are the same colour.
"""

from __future__ import annotations

from html import escape

# The surface this is drawn on, matching the tutoring workspace rather than
# being a second look. Contrast against --bg: ink 16.8:1, muted 8.6:1,
# faint 5.1:1, violet 5.4:1.
STYLE = """
  :root {
    --bg:      #0A0C14;
    --card:    #121626;
    --sunk:    #0E1220;
    --ink:     #EDEFF8;
    --muted:   #A7AECA;
    --faint:   #7B83A4;
    --line:    #222941;
    --violet:  #8B7CFF;
    --blue:    #4EA8FF;
    --held:    #5FD3A6;
    --broke:   #FF8F6E;
    --sans: "Outfit", "Segoe UI", system-ui, -apple-system, sans-serif;
    --mono: ui-monospace, "Cascadia Mono", Consolas, monospace;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 2rem 1.25rem 5rem;
    background: var(--bg); color: var(--ink);
    font-family: var(--sans); font-size: 15px; line-height: 1.6;
    -webkit-font-smoothing: antialiased;
  }
  .shell { max-width: 72rem; margin-inline: auto; }
  h1 { font-size: 1.5rem; letter-spacing: -0.02em; margin: 0 0 0.3rem; }
  h2 {
    font-size: 1.05rem; font-weight: 600; letter-spacing: -0.01em;
    margin: 2.4rem 0 0.2rem;
  }
  /* Every chart says what it is for. A dashboard whose panels need
     explaining out loud is one nobody reads without its author present. */
  .about { color: var(--faint); font-size: 0.88rem; margin: 0 0 0.9rem; max-width: 46rem; }
  a { color: var(--blue); }

  .filters { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 1rem 0 0.5rem; }
  .filters a, .filters b {
    padding: 0.3rem 0.8rem; border-radius: 999px; font-size: 0.85rem;
    border: 1px solid var(--line); text-decoration: none; color: var(--muted);
  }
  .filters b { background: var(--violet); color: #0A0C14; border-color: var(--violet); }

  .tiles {
    display: grid; gap: 0.8rem; margin-top: 1.4rem;
    grid-template-columns: repeat(auto-fit, minmax(10.5rem, 1fr));
  }
  .tile {
    background: var(--card); border: 1px solid var(--line);
    border-radius: 14px; padding: 1rem 1.1rem;
  }
  .tile .n {
    font-size: 2rem; font-weight: 600; letter-spacing: -0.03em; line-height: 1.1;
  }
  .tile .k { color: var(--faint); font-size: 0.8rem; margin-top: 0.2rem; }
  .tile .of { color: var(--muted); font-size: 0.85rem; font-weight: 400; }
  .tile.lead .n { color: var(--violet); }
  .tile.good .n { color: var(--held); }

  .panel {
    background: var(--card); border: 1px solid var(--line);
    border-radius: 16px; padding: 1.3rem 1.4rem; margin-top: 0.6rem;
  }
  .row2 { display: grid; gap: 0.8rem; grid-template-columns: 1fr 1fr; }
  @media (max-width: 760px) { .row2 { grid-template-columns: 1fr; } }

  /* A bar and its label on one line, label first, so the eye reads names
     down the left and lengths across - no rotated text anywhere. */
  .bars { display: grid; gap: 0.7rem; }
  .bar { display: grid; grid-template-columns: 13rem 1fr 5.5rem; gap: 0.8rem;
         align-items: center; }
  .bar .name { color: var(--muted); font-size: 0.88rem; min-width: 0; }
  .bar .name b { display: block; font-weight: 400; overflow: hidden;
                  text-overflow: ellipsis; white-space: nowrap; }
  /* The caption sits under the label, not beside the number: in the number
     column "page loads, not people" wrapped onto three lines and pushed the
     bars out of line with each other. */
  .bar .name i { font-style: normal; color: var(--faint); font-size: 0.78rem; }
  /* Both of these were spans, which are inline - so the height and the width
     did nothing at all and every bar rendered as an empty track. */
  .bar .track { display: block; background: var(--sunk); border-radius: 5px;
                height: 0.8rem; overflow: hidden; }
  .bar .fill { display: block; height: 100%; border-radius: 5px;
               background: var(--violet); min-width: 2px; }
  .bar .fill.blue { background: var(--blue); }
  .bar .fill.good { background: var(--held); }
  .bar .fill.bad  { background: var(--broke); }
  .bar .n { text-align: right; font-variant-numeric: tabular-nums;
             font-size: 1.05rem; font-weight: 600; white-space: nowrap; }
  @media (max-width: 560px) {
    .bar { grid-template-columns: 7.5rem 1fr 3.6rem; }
  }

  .empty { color: var(--faint); font-style: italic; }
  table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
  td { padding: 0.4rem 0.2rem; border-bottom: 1px solid var(--line); }
  td.n { text-align: right; font-variant-numeric: tabular-nums; color: var(--muted); }
  .foot { color: var(--faint); font-size: 0.85rem; margin-top: 2.5rem; }
"""


def rate(top: int, bottom: int) -> str:
    """A share, or a dash when there is nothing to take a share of.

    Printed rather than computed inline so that zero out of zero never comes
    out as 0%, which reads as failure rather than as silence.
    """
    if not bottom:
        return "&mdash;"
    return f"{top / bottom:.0%}"


def tile(number, label, *, of=None, kind="") -> str:
    inside = f'{number}<span class="of"> / {of}</span>' if of is not None else str(number)
    return (f'<div class="tile {kind}"><div class="n">{inside}</div>'
            f'<div class="k">{escape(label)}</div></div>')


def bars(rows, *, colour="", note=None) -> str:
    """Horizontal bars: (label, value, caption).

    Scaled against the largest, which is the only honest scale when the thing
    being compared is counts of the same kind.
    """
    rows = [r for r in rows if r is not None]
    if not rows:
        return f'<p class="empty">{escape(note or "Nothing yet.")}</p>'

    top = max(value for _, value, _ in rows) or 1
    out = []
    for label, value, caption in rows:
        width = round(100 * value / top, 1)
        # A real but tiny number still gets a sliver: the difference between
        # one person and none is the whole point of looking.
        width = max(width, 1.5) if value else 0
        under = f'<i>{escape(caption)}</i>' if caption else ''
        out.append(
            f'<div class="bar"><span class="name" title="{escape(str(label))}">'
            f'<b>{escape(str(label))}</b>{under}</span>'
            f'<span class="track"><span class="fill {colour}" '
            f'style="width:{width}%"></span></span>'
            f'<span class="n">{value}</span></div>')
    return f'<div class="bars">{"".join(out)}</div>'


def columns(days, key, *, colour="#8B7CFF", label="") -> str:
    """A day per column, oldest left. Empty days are drawn, not skipped."""
    top = max([row[key] for row in days] or [0])
    if not top:
        return '<p class="empty">Nothing in this window yet.</p>'

    width, gap, height = 14, 5, 74
    marks = []
    for i, row in enumerate(days):
        tall = max(round(height * row[key] / top), 2) if row[key] else 0
        marks.append(
            f'<rect x="{i * (width + gap)}" y="{height - tall}" width="{width}" '
            f'height="{tall}" rx="3" fill="{colour}">'
            f'<title>{escape(row["day"])}: {row[key]}</title></rect>')
    span = len(days) * (width + gap) - gap
    return (f'<svg viewBox="0 0 {span} {height}" width="100%" height="{height}" '
            f'role="img" aria-label="{escape(label)}, {len(days)} days, peak {top}">'
            f'{"".join(marks)}</svg>'
            f'<p class="about" style="margin:0.5rem 0 0">'
            f'{escape(days[0]["day"])} to {escape(days[-1]["day"])} '
            f'&middot; peak {top}</p>')


def funnel(steps) -> str:
    """Every step as a share of the one above it.

    The centrepiece. Counts alone cannot tell you where people leave, because
    a small number at the bottom might be a small number at the top - so each
    row carries what it kept from the step before.
    """
    rows = []
    top = steps[0][1] if steps else 0
    for i, (label, value, note) in enumerate(steps):
        before = steps[i - 1][1] if i else None
        if before:
            # Can happen honestly - a page can be opened more often than the
            # one before it - and saying so beats printing 140%.
            kept = ("more than the step above" if value > before
                    else f"{value / before:.0%} of the step above")
        elif i:
            # Nobody reached the step above, so there is no share to take.
            # Falling back to "of all" here printed 20% under a row showing
            # zero, which reads as a contradiction.
            kept = "nobody reached the step above"
        else:
            kept = note or ""
        rows.append((label, value, kept))
    return bars(rows)


def page(*, title, body, days, stage, stages) -> str:
    def when(n, text):
        here = (n == days)
        link = "?" + "&".join(
            bit for bit in (f"days={n}" if n else "", f"stage={stage}" if stage else "")
            if bit)
        return f"<b>{text}</b>" if here else f'<a href="{link or "?"}">{text}</a>'

    def by_stage(value, text):
        here = (value == stage)
        link = "?" + "&".join(
            bit for bit in (f"days={days}" if days else "", f"stage={value}" if value else "")
            if bit)
        return f"<b>{text}</b>" if here else f'<a href="{link or "?"}">{text}</a>'

    return f"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>{escape(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet"
      href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&display=swap">
<style>{STYLE}</style>
<div class="shell">
  <h1>{escape(title)}</h1>
  <p class="about">Everything here is counted on the server. Nothing identifies
  anybody: a "person" is one browser on one day, and the marker that makes that
  possible is replaced every night.</p>

  <div class="filters">
    {when(7, "7 days")} {when(30, "30 days")} {when(None, "All time")}
  </div>
  <div class="filters">
    {by_stage(None, "Everything")}
    {" ".join(by_stage(s, s.upper() if s == "gcse" else "A-level") for s in stages)}
    <a href="/admin/feedback">What people said &rarr;</a>
  </div>

  {body}

  <p class="foot">ownIT &middot; counts only, no personal data.</p>
</div>
"""
