# OwnIt

## v1 (on `main`)

A FastAPI backend that checks whether a student can account for an essay they submitted. Upload a PDF, get Socratic questions about the argument's load-bearing claims, answer them, get follow-ups that probe what the answers left unexplained, and produce a neutral observation report for the teacher. The report deliberately does not judge or score — it records what the student said, with quotes.

## v2 — Diagnostic (this branch)

v2 changes the subject and the shape of the problem.

**Input:** an A-level maths question a student is stuck on, plus their attempt (however partial, however wrong).

**Output:** the specific prerequisite skill that is actually missing.

### The idea

"I don't get integration by parts" is almost never the real problem. Underneath it might be a shaky grip on the product rule, or on rearranging an equation, or on what a definite integral means. A student who is told to re-read the integration-by-parts chapter will fail again, because the thing that broke was three levels down.

v2 works backwards from the failure. It takes the attempt, finds the first step where the reasoning actually goes wrong, and then asks: what must be true for a student to get that step right? That gives a prerequisite. If the student can do the prerequisite, the diagnosis stops there. If they can't, the same question is asked of *that* skill, and the trace goes one level deeper.

The result is a chain, not a label:

```
integration by parts
  └─ product rule for differentiation
       └─ recognising a product vs. a composition
```

with the bottom of the chain being the thing to actually teach.

### How it differs from v1

| | v1 | v2 |
|---|---|---|
| Domain | Essays | A-level maths |
| Asks about | Work the student already produced | Work the student is stuck on |
| Goal | Can they account for it? | What is missing underneath? |
| Output | Neutral observation report | A prerequisite chain and a single root gap |

What carries over is the method: ask, don't tell. The diagnostic never supplies the answer or works the problem for the student. Probing questions are how it tests whether a prerequisite is held, and the student's replies are the evidence.

### Open questions

- Where does the prerequisite map come from — is it inferred per-question, or is there a fixed A-level skill graph to walk?
- When does the trace stop? Too shallow and it restates the symptom; too deep and every diagnosis bottoms out at arithmetic.
- How many probing questions can we ask before a stuck student gives up?
- Does the report go to the student, the teacher, or both — and does it read differently for each?

### Status

Design only. No v2 code yet. `backend/main.py` is unchanged v1.
