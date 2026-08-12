from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from anthropic import Anthropic
from dotenv import load_dotenv
import io
from pypdf import PdfReader
import json


# This line reads the .env file and loads ANTHROPIC_API_KEY
# into memory, so the Anthropic() connection below can find it.
load_dotenv()

# This creates the API application. Everything hangs off this "app" object.
app = FastAPI(title="OwnIt API")

# This creates a connection to Claude's API. It reads the key from
# the ANTHROPIC_API_KEY environment variable automatically.
claude = Anthropic()


# This is a "route" - it says: when someone visits "/" with a GET request,
# run this function and send back whatever it returns.
@app.get("/")
def health_check():
    return {"status": "OwnIt backend is running"}


# This defines WHAT the frontend must send us: just one field, "essay_text".

class EssayInput(BaseModel):
    essay_text: str


# THIS is "the function" you asked about. YOU wrote it. It lives right here,
# in main.py. FastAPI's job is just to call it whenever a POST request
# arrives at "/generate-questions".
@app.post("/generate-questions")
def generate_questions(data: EssayInput):
    # data.essay_text is the PDF text the frontend sent us.
    # We're building a prompt and sending it to Claude.
    response = claude.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": (
                    "You are checking whether a student can account for their own "
    "written work. Generate 3 questions about the essay below.\n\n"
    "Rules:\n"
    "- Target the claims the argument depends on, not incidental "
    "facts, dates, or names that happen to appear.\n"
    "- Never ask something answerable by reading a single sentence "
    "of the essay. If the answer is sitting in the text, it is a "
    "bad question.\n"
    "- Ask about reasoning the student asserted but did not spell "
    "out. Force them to supply the thinking behind their own claim.\n"
    "- Stay within what the essay actually covers. Do not introduce "
    "counterarguments, alternative theories, or outside knowledge the "
    "student never raised. Every question must be answerable by "
    "someone who genuinely wrote this specific essay.\n\n"
    "Return only a numbered list, nothing else.\n\n"
    f"ESSAY:\n{data.essay_text}"
                ),
            }
        ],
    )


    # response.content[0].text is Claude's reply, as plain text.
    questions_text = response.content[0].text

    # We package it into a dictionary. FastAPI automatically converts
    # this into JSON before sending it back to the frontend.
    return {"questions": questions_text}


class FollowupInput(BaseModel):
    essay_text: str
    question: str
    student_answer: str


@app.post("/generate-followup")
def generate_followup(data: FollowupInput):
    response = claude.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": (
                    "A student submitted the essay below, was asked a question "
                    "about it, and gave the answer shown. Write ONE follow-up "
                    "question.\n\n"
                    "Rules:\n"
                    "- Identify what the answer leaves unexplained or assumes "
                    "without justification.\n"
                    "- If the answer depends on a more basic concept the student "
                    "has not shown they hold, probe that concept instead.\n"
                    "- If the answer contradicts or misrepresents what the essay "
                    "actually says, probe that directly.\n"
                    "- Do not teach, explain, or hint at the answer.\n"
                    "- Stay within the essay's own territory. No outside knowledge.\n\n"
                    "Return only the question, nothing else.\n\n"
                    f"ESSAY:\n{data.essay_text}\n\n"
                    f"QUESTION ASKED:\n{data.question}\n\n"
                    f"STUDENT ANSWER:\n{data.student_answer}"
                ),
            }
        ],
    )

    return {"followup": response.content[0].text}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # Read the uploaded file's raw bytes into memory
    contents = await file.read()

    # pypdf needs a file-like object, so we wrap the bytes
    reader = PdfReader(io.BytesIO(contents))

    # Loop through every page and collect its text
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    # If we got nothing, it's probably a scanned image PDF
    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No text found. This may be a scanned PDF, which isn't supported yet.",
        )

    return {"essay_text": text}

class ExchangeInput(BaseModel):
    question: str
    answer: str
    followup: str
    followup_answer: str


class ScoringInput(BaseModel):
    essay_text: str
    exchanges: list[ExchangeInput]



@app.post("/generate-report")
def generate_report(data: ScoringInput):
    # Turn the list of exchanges into readable text for the prompt
    transcript = ""
    for i, exchange in enumerate(data.exchanges, start=1):
        transcript += (
            f"\nQuestion {i}: {exchange.question}\n"
            f"Answer: {exchange.answer}\n"
            f"Follow-up: {exchange.followup}\n"
            f"Follow-up Answer: {exchange.followup_answer}\n"
        )

    response = claude.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": (
                    "A student submitted the essay below and was asked questions "
                    "about it. Report what happened, for their teacher.\n\n"
                    "Rules:\n"
                    "- Note where the student explained their reasoning in their "
                    "own terms, including anything they articulated more clearly "
                    "than the essay itself does.\n"
                    "- Note where answers restated the essay without adding to it, "
                    "or did not address what was asked.\n"
                    "- Quote the student's own words as evidence for each point.\n"
                    "- Do NOT judge, conclude, or state whether the student wrote "
                    "the essay. Do NOT score or rate. Report only what they said.\n\n"
                    "Return ONLY this JSON, nothing else:\n"
                    '{"accounted_for": [], "gaps": [], "summary": ""}\n\n'
                    "accounted_for: observations of what they explained in their "
                    "own terms, each with a quote.\n"
                    "gaps: observations of where answers stayed thin or missed the "
                    "question, each with a quote.\n"
                    "summary: two or three neutral sentences describing the "
                    "session.\n\n"
                    f"ESSAY:\n{data.essay_text}\n\n"
                    f"TRANSCRIPT:\n{transcript}"
                ),
            }
        ],
    )

    raw = response.content[0].text.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    return json.loads(raw[start : end + 1])