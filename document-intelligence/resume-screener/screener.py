from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScreenResult:
    """Structured screening output for one resume against one job description.

    Scores ONLY job-relevant skills and experience - never demographic signals."""

    score: int  # 0-100 skills/experience fit
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    years_experience: Optional[float] = None
    recommendation: str = ""  # advance | maybe | reject
    rationale: str = ""

    def band(self) -> str:
        if self.score >= 70:
            return "advance"
        if self.score >= 45:
            return "maybe"
        return "reject"


_WORD = re.compile(r"[a-z0-9+#.]+")

# Words that signal a required skill in a JD; we extract the noun-phrase after them.
_REQ_HINT = re.compile(
    r"\b(experience with|proficient in|knowledge of|skilled in|familiar with|expertise in|using)\s+",
    re.I,
)


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def extract_required_skills(job_desc: str) -> list[str]:
    """Pull a candidate skill list from a JD: bullet/comma items + 'experience with X' phrases.

    A heuristic - a recruiter should review/edit the list before screening at scale."""
    skills: list[str] = []
    # explicit "Skills: a, b, c" or bulleted requirements
    for line in job_desc.splitlines():
        line = line.strip(" -*\t")
        m = re.match(r"(?:required skills|skills|requirements|must have)\s*:\s*(.+)", line, re.I)
        if m:
            skills += [s.strip() for s in re.split(r"[,;/]", m.group(1)) if s.strip()]
    # inline "experience with X" phrases
    for m in _REQ_HINT.finditer(job_desc):
        tail = job_desc[m.end():m.end() + 40]
        phrase = re.split(r"[,.;\n]", tail)[0].strip()
        if 1 <= len(phrase.split()) <= 4:
            skills.append(phrase)
    # dedupe, keep order, drop noise
    seen, out = set(), []
    for s in skills:
        key = s.lower()
        if key and key not in seen and len(key) > 1:
            seen.add(key)
            out.append(s)
    return out[:15]


def extract_years(resume: str) -> Optional[float]:
    """Best-effort years-of-experience: explicit 'X years' beats nothing."""
    m = re.search(r"(\d{1,2})\+?\s*years?(?:\s+of)?\s+(?:experience|exp)", resume, re.I)
    return float(m.group(1)) if m else None


def heuristic_screen(resume: str, job_desc: str) -> ScreenResult:
    """Score a resume on skill overlap + years, with no LLM. Deterministic and auditable.

    Scoring: 80% weight on required-skill coverage, 20% on a years-of-experience signal."""
    required = extract_required_skills(job_desc)
    resume_tokens = _tokens(resume)

    matched, missing = [], []
    for skill in required:
        skill_tokens = _tokens(skill)
        # a skill counts as matched if all its significant tokens appear in the resume
        if skill_tokens and skill_tokens <= resume_tokens:
            matched.append(skill)
        else:
            missing.append(skill)

    skill_score = (len(matched) / len(required)) if required else 0.0
    years = extract_years(resume)
    years_score = min(years / 5.0, 1.0) if years else 0.0  # cap credit at 5 yrs

    score = round(100 * (0.8 * skill_score + 0.2 * years_score))
    result = ScreenResult(
        score=score,
        matched_skills=matched,
        missing_skills=missing,
        years_experience=years,
        rationale=(
            f"Matched {len(matched)}/{len(required)} required skills"
            + (f"; {years:.0f} yrs experience detected." if years else "; no explicit years found.")
        ),
    )
    result.recommendation = result.band()
    return result


def llm_screen(resume: str, job_desc: str, api_key: str) -> ScreenResult:
    """Use Claude to score fit against the JD with a rubric. Skills/experience only."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1200,
        messages=[
            {
                "role": "user",
                "content": (
                    "You are screening a resume against a job description. Score ONLY "
                    "job-relevant skills and experience. Do NOT consider or infer name, gender, "
                    "age, ethnicity, nationality, or any protected attribute. "
                    "Respond with ONLY valid JSON, no markdown fences:\n"
                    '{"score": 0-100, "matched_skills": ["..."], "missing_skills": ["..."], '
                    '"years_experience": number_or_null, "recommendation": "advance|maybe|reject", '
                    '"rationale": "2-3 sentences on skills fit"}\n\n'
                    f"JOB DESCRIPTION:\n{job_desc}\n\nRESUME:\n{resume}"
                ),
            }
        ],
    )
    data = json.loads(response.content[0].text.strip())
    return ScreenResult(
        score=int(data.get("score", 0)),
        matched_skills=data.get("matched_skills", []),
        missing_skills=data.get("missing_skills", []),
        years_experience=data.get("years_experience"),
        recommendation=data.get("recommendation", ""),
        rationale=data.get("rationale", ""),
    )


def screen_resume(resume: str, job_desc: str, api_key: Optional[str] = None) -> ScreenResult:
    """Screen a resume against a job description.

    Uses Claude if an API key is available, else deterministic skill-overlap scoring."""
    if not resume.strip() or not job_desc.strip():
        return ScreenResult(score=0, rationale="Missing resume or job description.")
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return heuristic_screen(resume, job_desc)
    return llm_screen(resume, job_desc, api_key)


SAMPLE_JOB = """\
Senior Data Engineer

We are hiring a Senior Data Engineer to own our analytics platform.

Required skills: Python, SQL, Airflow, dbt, AWS, Snowflake

Responsibilities:
- Build and maintain ETL pipelines with experience with Spark
- Strong knowledge of data modeling and warehouse design
- 5+ years of experience in data engineering
"""

SAMPLE_RESUMES = {
    "Strong fit": """\
Data engineer with 6 years of experience building data platforms.
Skills: Python, SQL, Airflow, dbt, AWS, Snowflake, Spark, Terraform.
Built ETL pipelines processing 2TB/day and led data modeling for the warehouse.
""",
    "Partial fit": """\
Analytics engineer, 3 years of experience.
Skills: SQL, dbt, Python, Looker. Some exposure to AWS.
Built reporting models and dashboards for the marketing team.
""",
    "Weak fit": """\
Frontend developer with 4 years experience.
Skills: JavaScript, React, CSS, Figma. Built customer-facing web apps.
""",
}
