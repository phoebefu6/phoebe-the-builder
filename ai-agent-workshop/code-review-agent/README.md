# Code Review Agent

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phoebefu6/phoebe-the-builder/blob/main/ai-agent-workshop/code-review-agent/demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/phoebefu6/phoebe-the-builder/main?labpath=ai-agent-workshop/code-review-agent/demo.ipynb)

> "PRs sit unreviewed for days" — an agent that gives every PR an instant first pass so reviewers focus on logic, not nitpicks.

## Business Impact
- **Before:** PRs wait in the queue while reviewers manually check for secrets, bare excepts, debug leftovers, TODOs
- **After:** agent flags mechanical issues (secrets, `except:`, leftover `print`/`pdb`, TODOs, long lines) and writes a verdict instantly
- **Estimated ROI:** hours/week of reviewer time redirected from nitpicks to actual logic review

## Tech Stack
Python, Claude API (Anthropic SDK), GitHub API (diff fetch), Streamlit, Docker

## Demo

**[Run the interactive demo notebook →](demo.ipynb)** — pre-rendered with outputs, or click the Colab/Binder badges above to run it live.

For the Streamlit app:
```bash
pip install -r requirements.txt
streamlit run app.py
```

Set `ANTHROPIC_API_KEY` for a Claude-written review summary (falls back to a rule-based summary otherwise). Set `GITHUB_TOKEN` to fetch diffs from private PRs via the GitHub API.

## Learning Connection
Built while studying Agentic AI and GitHub Actions for CI/CD on LinkedIn Learning.
Applies: automated first-pass review as a gate before human review, same pattern used in CI-integrated review bots.

## Impact Note
- **Who benefits:** engineering teams with a PR backlog and limited reviewer bandwidth
- **Potential risks:** rule-based findings can false-positive (e.g. flagging a test fixture secret); agent is a first pass, not a replacement for human review, especially on logic/architecture
