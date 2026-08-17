# OwnIt

Two things live in this repository.

**v2 — the maths diagnostic** (this branch). A student is stuck on a question.
They send it in, and instead of being told how to do it, they are asked a short
run of questions that traces the failure downwards through the skills the
question rests on, until it reaches the one that is actually missing.

**v1 — the essay checker** (`backend/main.py`). The original: upload an essay,
answer questions about it, and a teacher gets a neutral report of what the
student could and could not account for. Untouched by v2 except for a model
migration.

---

## v2, in one pass

```
    a photo of a question
            |
    entry.py     which skill is this asking for?  (or: we cannot place it)
            |
    walk.py      ask about a prerequisite. Held? try the next one.
            |    Not held? go down into it and ask again.
            |
    the gap      the deepest skill they cannot do whose own
                 prerequisites they can
```

The rest is in service of that:

| | |
|---|---|
| `data/skills.yaml` | 72 skills and what each rests on. 24 are doorways a question can start from |
| `backend/graph.py` | loads it, refuses to run on a broken graph |
| `backend/questions.py` | writes one multiple-choice question about one skill |
| `backend/bank.py` | keeps questions so they are written once per skill, not once per student |
| `backend/entry.py` | question in - photo, PDF or text - to a skill to start from |
| `backend/walk.py` | the descent |
| `backend/store.py` | every answer, every gap, every question we could not place |
| `backend/api.py` | the same thing over HTTP, one question per request |
| `web/index.html` | the page |
| `backend/coverage.py` | how many real questions can we place? |
| `backend/scheme.py` | what do mark schemes say a question actually needs? |
| `backend/llm.py` | which model each job gets |

## Running it

```powershell
pip install -r requirements.txt

# the website
uvicorn api:app --reload --port 8000 --app-dir backend

# or from a terminal, no server
python backend/entry.py "Find the stationary points of y = x^3 - 6x^2 + 9x" --role tutor
python backend/entry.py "" --image data/papers/question.png

# how much of a real paper can we handle?
python backend/coverage.py data/papers/paper.pdf

# what does a mark scheme say questions need?
python backend/scheme.py data/papers/markscheme.pdf
```

`ANTHROPIC_API_KEY` goes in `backend/.env`.

```powershell
python -m pytest backend/ -q      # 170 tests, none of which call the API
```

## Deploying

The application is stateless; everything is in one SQLite file.

```
web: uvicorn api:app --host 0.0.0.0 --port $PORT --app-dir backend
```

Two environment variables:

- `ANTHROPIC_API_KEY`
- `OWNIT_DB` — where the database lives

**`OWNIT_DB` matters more than it looks.** Hosting platforms wipe the
filesystem on every deploy, so it has to point at a disk that survives -
otherwise every session, every misconception and every banked question
disappears the next time you push. That data is the only part of this that a
competitor could not rebuild from the code.

One instance only, while it is SQLite. Several would need Postgres, and nothing
above would change except the connection.

## Where it actually stands

**Working, and checked against real questions:** reading a photo or a PDF of a
question, matching it to a skill or honestly refusing, narrowing from a
student's attempt, the descent itself, the question bank, and the record of
everything answered.

**Coverage is 20%** of a real Edexcel Pure paper - 3 questions in 15. The
backlog is not a guess: `coverage.py` and `scheme.py` produce it from actual
papers. Biggest gaps are integration, trigonometry, modulus functions,
composite and inverse functions, binomial expansion and series.

**Not yet done.** Nothing measures whether a diagnosis is *correct* -
`backend/eval.py` is still empty, and that is the gap that matters most before
real students see it. Nobody has corrected a diagnosis yet, so the `feedback`
table is empty. And the report is written for a tutor; the same finding needs
gentler words before it is put in front of a sixteen-year-old.
