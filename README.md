# ownIT

A student is stuck on an A-level maths question. ownIT does not explain the
question, and it does not re-teach the topic. It works downwards through what
that question rests on — one short question at a time — until it reaches the
thing that is actually missing.

The premise is that the topic is rarely the problem. A student who cannot
differentiate is usually not missing differentiation; they are missing
something two years underneath it that nobody has ever gone looking for,
because asking "what don't you understand?" only works on people who already
know.

**In active beta.** Live at
[ownit-tzsv.onrender.com](https://ownit-tzsv.onrender.com), written for tutors,
with a waitlist open. Nobody has used it in a lesson yet.

---

## The idea, in one worked example

AQA Paper 1, June 2024, question 10 asks students to prove from first
principles that the derivative of `x³` is `3x²`. The examiner report says a
number of them expanded `(x + h)³` wrongly and never collected like terms.

Every one of those students lost marks on a *differentiation* question. Not one
of them had a differentiation problem — they could not expand a bracket,
something they were taught at fourteen. ownIT finds that in four questions:

```
Prove from first principles…
  ├─ Limits — what h → 0 means        held
  ├─ Algebraic fractions              held
  ├─ Expanding brackets               ← here
  └─   Negative number arithmetic     held
```

Everything underneath expanding brackets is solid, so that is where to start —
not lower, and not on differentiation.

## How it works

```
   a question — typed, a photo, or a PDF
            │
   entry.py │ which of the 49 doorways is this?  (or: we cannot place it)
            │ if it has parts (a)-(d), which one are you stuck on?
            │
   walk.py  │ ask about something it rests on. Held? try the next.
            │ Not held? go down into that and ask again.
            │
   the gap  │ the deepest skill they cannot do whose own
            │ prerequisites they can
```

**What you send it.** A question on its own is enough. A photo is better than
typing — copying maths into text destroys it, and there is proof:
`(1 − 9x)^½` arrived as `19 1 2 x` when pasted, and read perfectly from a
picture. Optionally, what the student tried, typed or photographed, which lets
the walk skip straight to the branch their working broke in.

**The rules.** Nearest-to-the-question first, so it feels connected to what they
came for. Five levels deep, because measuring it moved root-cause accuracy from
40% to 87%. Fifteen questions maximum, though the average is 5.9. "I don't know"
is always offered, counts as not-held, and is recorded separately from a wrong
answer — those are different students.

**What comes out.** One skill to teach, the reason it is that one and not
something lower, and an honest list of the branches that were not checked.

## The question bank

550 questions, covering all 108 skills, written once per skill and served to
many students — which is why a diagnosis costs 2–8p rather than paying to write
a question every time.

**Every one has been checked by a model that did not write it.** The checker is
never shown which answer is marked correct: it gets the question and all four
options, works it out, and picks one. Only then is its pick compared with the
stored key. Anything the two disagree on goes to a stronger model before being
retired.

That audit found **10% of the bank had the wrong answer marked correct** — 55 of
the first 531 — and not evenly: 40% in integration, 4% in the foundational
skills. The harder the arithmetic, the worse it got. A second pass over the
replacements found another 19 of 71. Generation moved from Haiku to Sonnet on
the back of a measured comparison (45% wrong keys against 0% on the worst
topics), and 77 questions are now retired.

This matters more than it sounds. A wrong answer key is the one failure the
diagnostic cannot see: a student answers *correctly*, is recorded as not
holding the skill, and the walk goes hunting a gap that is not there. The report
looks completely normal.

The bank travels as `data/question_bank.jsonl`, committed here, and refills an
empty database at startup — which matters because the host wipes the filesystem
on every deploy.

## What it covers today

**Roughly 60% of a real Edexcel Pure paper** can be placed at a doorway, up from
20% before five topics were added. Measured by `coverage.py` against actual
papers, not estimated.

| Strongest | Also covered |
|---|---|
| Differentiation — 13 doorways, several levels deep | Integration, binomial expansion, sequences and series, quadratics, functions, indices and surds, logs and exponentials, coordinate geometry, polynomials |

**The other 40% is honestly missing.** No trigonometry, circles, vectors, proof
or numerical methods — and no statistics or mechanics at all, which is roughly a
third of A-level Maths. A question it cannot place is refused and filed rather
than forced into the nearest doorway, because a wrong match sends a student down
a diagnosis built on the wrong question. Those refusals become the build list.

## Is it right?

`eval.py` invents students. It hides one skill, breaks everything resting on it,
runs the walk, and checks whether the hidden skill comes back — every skill under
every doorway, 718 runs.

```
found       619 / 708   (87%)   gaps within reach, named correctly
declined     10 / 10   (100%)   too deep to reach, correctly unconfirmed
false         0 / 718     (0%)   named a skill the student could do
questions           5.9 average, 13 at worst
```

Zero false positives is the one that matters. Missing a gap wastes a session;
naming a skill the student already has sends a tutor to teach the wrong thing
*and* stops anyone looking further.

This is not the same as testing on real students. That has not happened yet.

## Running it

```powershell
pip install -r requirements.txt

# the site — landing page at /, the diagnostic at /start
uvicorn api:app --reload --port 8000 --app-dir backend

# or from a terminal, no server
python backend/walk.py differentiate_function
python backend/entry.py "Find the stationary points of y = x^3 - 6x^2 + 9x"
python backend/entry.py "" --image data/papers/question.png

# the bank
python backend/bank.py                 # what is on the shelf
python backend/bank.py --save          # write it out to be committed
python backend/bank.py --dry           # skills that cost a live generation
python backend/audit.py                # check the answer keys (prices itself first)

# how much of a real paper can we handle?
python backend/coverage.py data/papers/paper.pdf
python backend/eval.py                 # is the diagnosis correct?
```

`ANTHROPIC_API_KEY` goes in `backend/.env`.

```powershell
python -m pytest backend/ -q      # 321 tests, none of which call the API
```

Tests cover the graph, the walk, the store, the API, the bank, the audit, the
maths rendering, and — since a comment containing a closing script tag once
printed the page's own source at students — the pages are parsed as HTML on
every run, because a JavaScript engine cannot see that class of bug.

## Deploying

`render.yaml` is a Blueprint: point Render at this repository.

**The disk is the whole point of that file.** Without it the filesystem goes on
every deploy and every restart, and SQLite goes with it — waitlist signups,
tutor verdicts on whether a diagnosis was right, and every question that could
not be placed. The bank would survive, being committed here; nothing else would.

| Variable | |
|---|---|
| `ANTHROPIC_API_KEY` | set by hand in the dashboard, never committed |
| `OWNIT_DB` | put it on the disk: `/var/data/ownit.db` |
| `OWNIT_ADMIN_PASSWORD` | opens the admin pages; with none set they refuse to open at all |
| `OWNIT_DAILY_STARTS` | how many questions may be read in a day. 30 by default |

`/api/health` answers two questions from outside, which matters because a disk
that failed to mount looks exactly like a working app until somebody notices
the waitlist is empty:

```json
{"storage": "kept", "questions": 550}
```

`storage: kept` means the disk mounted. `questions: 0` means the shelf did not
restock, and every question is about to cost money.

One instance while it is SQLite. Several would need Postgres, and nothing above
would change except the connection.

## Watching it

The page is public with no sign-in, and every question read is a call to the
best model, so there is a daily ceiling. It counts readings rather than walks,
because a reading is the thing that costs money — one question can be read once
and walked three times as someone moves between its parts.

| | |
|---|---|
| `/admin/numbers` | the funnel — opened, sent a question, started, finished — where people stop, and whether they said it helped |
| `/admin/feedback` | what tutors actually wrote: verdicts, corrections, and comments left without running a diagnosis |

Both are password-protected, both are `noindex`, and everything on them is
escaped, because it was typed into a form anyone on the internet can reach.

Two questions are asked at the end and kept apart on purpose. **"Was this any
use?"** is one tap from anybody. **"Was this right?"** is for tutors, and it takes
the real gap in their own words — matched to the graph when it is a skill we
know, kept verbatim when it is not, because "it was reading the question
properly" is the most useful thing a tutor could tell us.

## What is still missing

- **Nobody has used it.** Every accuracy number here comes from simulated
  students. The feedback tables work and are empty.
- **40% of a paper is unplaceable**, including all of statistics and mechanics.
- **Tutor mode is a flag, not a login.** `?tutor=1` keeps the verdict form and
  the running descent away from students; it stops nobody who edits a URL.
- **The API is open cross-origin.** The daily ceiling limits the damage rather
  than preventing it.

## Also in here

`backend/main.py` is v1 — the original essay checker. Upload an essay, answer
questions about it, and a teacher gets a neutral report of what the student
could and could not account for. Untouched by v2 except for a model migration,
and not part of the diagnostic.

`docs/inside-ownit.html` is a longer walkthrough of how all of this fits
together, written to be read rather than skimmed.
