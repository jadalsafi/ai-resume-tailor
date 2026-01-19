# AI Resume Tailor

A simple Python tool that takes a resume + job posting and generates a 
tailored version using the OpenAI API.

It outputs:
- `output.json` (structured results)
- `output.md` (readable formatted output)
- `output.pdf` (PDF export)

---

## Features
- Reads `resume.txt` and `job.txt`
- Uses OpenAI to tailor content based on the job posting
- Exports Markdown + PDF automatically
- Uses `.env` for secure API key storage

---

## Setup

### 1) Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/ai-resume-tailor.git
cd ai-resume-tailor

