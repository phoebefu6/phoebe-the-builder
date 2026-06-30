# Resume Screener

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/document-intelligence/resume-screener/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=document-intelligence/resume-screener/demo.ipynb)

> Rank a stack of resumes by skills fit — a triage aid, not a hiring decision.

## Business Impact
- **Before:** HR reads every resume cold — up to 40 hrs/week on first-pass screening.
- **After:** The job's required skills are extracted, each resume scored on coverage + experience, and the stack ranked — humans start at the top and spend time on judgement.
- **Estimated ROI:** ~20-30 hrs/week of first-pass reading redirected to real evaluation.

## ⚠️ Responsible use — read first
Resume screening is high-stakes and bias-prone. This tool:
- Scores **only job-relevant skills and experience** — never name, gender, age, ethnicity, school prestige, or any protected attribute.
- Uses **auditable** scoring (you can see exactly why a score is what it is).
- **Never auto-rejects.** It ranks; a human reviews every candidate, especially the "maybe" band.

## Tech Stack
Python · skill extraction + overlap scoring (deterministic, auditable) · Claude API (optional, rubric-scored) · Streamlit · matplotlib · Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Paste a job description and a batch of resumes (split by `===`). Get a ranked list with matched/missing skills per candidate. Set an `ANTHROPIC_API_KEY` to upgrade to Claude rubric scoring.

## How it works
1. **Extract** required skills from the JD (explicit "Required skills:" lines + inline "experience with X" phrases).
2. **Score** each resume: 80% required-skill coverage + 20% years-of-experience signal.
3. **Band** into advance (≥70) / maybe (45-69) / reject (<45) and rank.
4. **Claude mode** reads paraphrased evidence ("led data modeling" → warehouse-design skill) the keyword scorer misses, with the same skills-only, no-protected-attributes rubric.

## Learning Connection
Built while studying **Prompt Engineering — rubric & structured scoring** (Anthropic).
Applies: schema-constrained scoring, explicit bias guardrails in the prompt, and keeping a deterministic auditable fallback.

## Impact Note
- **Who benefits:** recruiters, hiring managers, small teams without an ATS.
- **Potential risks:** keyword scoring penalizes non-standard phrasing and career changers; an over-trusted score becomes a de-facto auto-reject. Mitigate by always reviewing the "maybe" band, editing the extracted skill list, and never wiring this to automatic rejection.
