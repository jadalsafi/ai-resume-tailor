import os
import json
import argparse
import re
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI

from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("Missing OPENAI_API_KEY.")
    print("Fix: create a .env file and add your key like this:")
    print("OPENAI_API_KEY=sk-...")
    raise SystemExit(1)

client = OpenAI(api_key=API_KEY)

STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","when","while","for","to","of","in","on","at","by","with",
    "is","are","was","were","be","been","being","as","it","this","that","these","those","from","into","over","under",
    "you","your","we","our","they","their","i","me","my","he","she","him","her","them","us",
    "will","would","can","could","should","may","might","must","do","does","did","done",
    "role","position","work","team","teams","experience","skills","skill","required","requirements","preferred",
    "including","include","ability","strong","excellent","good","knowledge","understanding",
    "using","use","used","based","within","across","etc"
}

def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def write_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()

def tokenize(s: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9\+\#\.\-]+", s.lower())
    cleaned = []
    for t in tokens:
        t = t.strip(".-")
        if not t:
            continue
        if t in STOPWORDS:
            continue
        if len(t) <= 2:
            continue
        cleaned.append(t)
    return cleaned

def keyword_candidates(text: str) -> Counter:
    tokens = tokenize(text)
    tech_bonus = {
        "python", "c", "c++", "linux", "git", "sql", "api", "docker",
        "automation", "testing", "embedded", "microcontroller"
    }
    counts = Counter(tokens)
    for k in list(counts.keys()):
        if k in tech_bonus:
            counts[k] += 2
    return counts

def top_keywords(job_text: str, n: int = 18) -> list[str]:
    counts = keyword_candidates(job_text)
    return [w for w, _ in counts.most_common(n)]

def match_score(resume_text: str, job_text: str) -> dict:
    job_keys = top_keywords(job_text, n=18)
    resume_norm = normalize_text(resume_text)

    present = [k for k in job_keys if k in resume_norm]
    missing = [k for k in job_keys if k not in resume_norm]

    coverage = (len(present) / max(1, len(job_keys))) * 100.0

    jr = set(tokenize(resume_text))
    jj = set(tokenize(job_text))
    jaccard = (len(jr & jj) / max(1, len(jr | jj))) * 100.0

    blended = 0.75 * coverage + 0.25 * jaccard

    return {
        "match_percent": round(blended, 1),
        "coverage_percent": round(coverage, 1),
        "jaccard_percent": round(jaccard, 1),
        "job_top_keywords": job_keys,
        "keywords_present": present,
        "missing_keywords": missing
    }

def build_prompt(resume_text: str, job_text: str, mode: str) -> str:
    requested = {
        "all": "summary, keywords, bullets, cover_letter",
        "summary": "summary",
        "keywords": "keywords",
        "bullets": "bullets",
        "coverletter": "cover_letter",
    }[mode]

    return f"""
You are an expert technical recruiter and resume writer.

TASK:
Given the RESUME and JOB POSTING, produce ONLY the following fields:
{requested}

OUTPUT FORMAT:
Return valid JSON ONLY (no markdown).

Keys:
- summary (string, 3 lines max)
- keywords (array of 6 strings)
- bullets (array of 4 strings)
- cover_letter (string, 150-200 words)

RULES:
- Do NOT invent experience.
- Keep it ATS-friendly.
- Use strong action verbs.

RESUME:
{resume_text}

JOB POSTING:
{job_text}
""".strip()

def call_model(prompt: str, model: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    text = resp.choices[0].message.content.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
        raise

def to_markdown(data: dict, score_block: dict | None = None) -> str:
    parts = []

    if score_block:
        parts.append("## Match Score\n")
        parts.append("- Match % (blended): " + str(score_block["match_percent"]) + "%")
        parts.append("- Keyword coverage: " + str(score_block["coverage_percent"]) + "%")
        parts.append("- Token similarity: " + str(score_block["jaccard_percent"]) + "%\n")

        parts.append("### Missing Keywords (from job posting)\n")
        if score_block["missing_keywords"]:
            parts.append(", ".join(score_block["missing_keywords"]) + "\n")
        else:
            parts.append("None\n")

    if "summary" in data:
        parts.append("## Tailored Summary\n")
        parts.append(data["summary"].strip() + "\n")

    if "keywords" in data:
        parts.append("## ATS Keywords (suggested)\n")
        parts.append(", ".join(data["keywords"]) + "\n")

    if "bullets" in data:
        parts.append("## Improved Resume Bullets\n")
        for b in data["bullets"]:
            parts.append("- " + b)
        parts.append("")

    if "cover_letter" in data:
        parts.append("## Cover Letter\n")
        parts.append(data["cover_letter"].strip() + "\n")

    return "\n".join(parts).strip() + "\n"

def markdown_to_plain(md: str) -> str:
    text = md
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = text.replace("**", "")
    text = text.replace("`", "")
    return text

def export_pdf(text: str, out_pdf: str) -> None:
    c = canvas.Canvas(out_pdf, pagesize=letter)
    width, height = letter
    left = 54
    top = height - 54
    line_height = 14
    max_width = width - 2 * left

    def wrap_line(line: str) -> list[str]:
        words = line.split()
        if not words:
            return [""]
        lines = []
        cur = words[0]
        for w in words[1:]:
            test = cur + " " + w
            if stringWidth(test, "Helvetica", 11) <= max_width:
                cur = test
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        return lines

    y = top
    c.setFont("Helvetica", 11)

    for raw in text.splitlines():
        for line in wrap_line(raw):
            if y < 54:
                c.showPage()
                c.setFont("Helvetica", 11)
                y = top
            c.drawString(left, y, line)
            y -= line_height

    c.save()

def main():
    parser = argparse.ArgumentParser(description="AI Resume + Job Tailor (OpenAI API)")
    parser.add_argument("--mode", choices=["all", "summary", "keywords", "bullets", "coverletter"], default="all")
    parser.add_argument("--resume", default="resume.txt")
    parser.add_argument("--job", default="job.txt")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--out_md", default="output.md")
    parser.add_argument("--out_json", default="output.json")
    parser.add_argument("--out_pdf", default="output.pdf")
    parser.add_argument("--no_pdf", action="store_true")

    args = parser.parse_args()

    resume_text = read_file(args.resume)
    job_text = read_file(args.job)

    score = match_score(resume_text, job_text)

    prompt = build_prompt(resume_text, job_text, args.mode)
    data = call_model(prompt, args.model)

    data["_match_analysis"] = score

    write_json(args.out_json, data)
    md = to_markdown(data, score_block=score)
    write_text(args.out_md, md)

    if not args.no_pdf:
        plain = markdown_to_plain(md)
        export_pdf(plain, args.out_pdf)

    print("Done!")
    print("JSON:", args.out_json)
    print("MD:", args.out_md)
    if not args.no_pdf:
        print("PDF:", args.out_pdf)

if __name__ == "__main__":
    main()
