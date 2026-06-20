from __future__ import annotations

"""Core logic: parse heterogeneous log lines, classify severity, and evaluate
alert rules.

Teams grep through logs by hand and miss the signal. This module turns raw lines
from several common formats (JSON, syslog, generic `LEVEL` lines, Apache/nginx
access logs) into structured records, then runs a small rule engine so an error
spike pages someone instead of scrolling past.

Designed to be the parsing core behind the Streamlit app *and* a future
"Observability" app on the platform shell - pure functions, no UI, no globals.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

LEVELS = ["DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL", "FATAL"]
SEVERITY = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "WARN": 30, "ERROR": 40, "CRITICAL": 50, "FATAL": 50}

# Generic "timestamp LEVEL message" - e.g. "2026-06-20 10:15:01 ERROR db timeout"
_GENERIC = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)\s+"
    r"(?P<level>" + "|".join(LEVELS) + r")\b[:\s]+(?P<msg>.*)$",
    re.IGNORECASE,
)
# Apache/nginx common log format - status code drives severity.
_ACCESS = re.compile(
    r'^(?P<ip>\S+)\s+\S+\s+\S+\s+\[(?P<ts>[^\]]+)\]\s+"(?P<req>[^"]*)"\s+(?P<status>\d{3})\s+(?P<size>\S+)'
)


@dataclass
class LogRecord:
    raw: str
    level: str = "UNKNOWN"
    severity: int = 0
    timestamp: Optional[str] = None
    message: str = ""
    source_format: str = "unparsed"
    fields: Dict[str, str] = field(default_factory=dict)


def _level_from_status(status: int) -> str:
    if status >= 500:
        return "ERROR"
    if status >= 400:
        return "WARNING"
    return "INFO"


def parse_line(line: str) -> LogRecord:
    """Parse a single log line, trying each known format in turn."""
    line = line.rstrip("\n")
    if not line.strip():
        return LogRecord(raw=line, source_format="blank")

    # 1. JSON logs.
    stripped = line.strip()
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
            level = str(obj.get("level") or obj.get("severity") or "INFO").upper()
            return LogRecord(
                raw=line,
                level=level if level in SEVERITY else "INFO",
                severity=SEVERITY.get(level, 20),
                timestamp=str(obj.get("timestamp") or obj.get("time") or obj.get("ts") or "") or None,
                message=str(obj.get("message") or obj.get("msg") or stripped),
                source_format="json",
                fields={k: str(v) for k, v in obj.items()},
            )
        except json.JSONDecodeError:
            pass

    # 2. Generic timestamped LEVEL line.
    m = _GENERIC.match(line)
    if m:
        level = m.group("level").upper()
        return LogRecord(
            raw=line,
            level=level,
            severity=SEVERITY.get(level, 0),
            timestamp=m.group("ts"),
            message=m.group("msg").strip(),
            source_format="generic",
        )

    # 3. Access log - severity from HTTP status.
    m = _ACCESS.match(line)
    if m:
        status = int(m.group("status"))
        level = _level_from_status(status)
        return LogRecord(
            raw=line,
            level=level,
            severity=SEVERITY[level],
            timestamp=m.group("ts"),
            message=f'{m.group("req")} -> {status}',
            source_format="access",
            fields={"ip": m.group("ip"), "status": str(status)},
        )

    # 4. Fallback - scan for any level keyword.
    upper = line.upper()
    for lv in sorted(SEVERITY, key=lambda x: -SEVERITY[x]):
        if re.search(rf"\b{lv}\b", upper):
            return LogRecord(raw=line, level=lv, severity=SEVERITY[lv], message=line.strip(), source_format="keyword")
    return LogRecord(raw=line, level="UNKNOWN", severity=0, message=line.strip(), source_format="unparsed")


def parse_lines(lines: List[str]) -> List[LogRecord]:
    return [parse_line(ln) for ln in lines if ln.strip()]


def summarize(records: List[LogRecord]) -> Dict[str, object]:
    """Aggregate counts by level + the most frequent error messages."""
    by_level: Dict[str, int] = {}
    error_msgs: Dict[str, int] = {}
    for r in records:
        by_level[r.level] = by_level.get(r.level, 0) + 1
        if r.severity >= SEVERITY["ERROR"]:
            error_msgs[r.message] = error_msgs.get(r.message, 0) + 1
    top_errors = sorted(error_msgs.items(), key=lambda x: -x[1])[:10]
    return {
        "total": len(records),
        "by_level": dict(sorted(by_level.items(), key=lambda x: -SEVERITY.get(x[0], 0))),
        "error_count": sum(1 for r in records if r.severity >= SEVERITY["ERROR"]),
        "top_errors": top_errors,
    }


# --- Alert rule engine ---------------------------------------------------------

@dataclass
class AlertRule:
    name: str
    min_level: str = "ERROR"          # fire on records at/above this severity
    threshold: int = 1                # how many matching records trigger the alert
    contains: Optional[str] = None    # optional substring the message must contain


def evaluate_rules(records: List[LogRecord], rules: List[AlertRule]) -> List[Dict[str, object]]:
    """Return one result per rule with the match count and fired/quiet state."""
    out: List[Dict[str, object]] = []
    for rule in rules:
        floor = SEVERITY.get(rule.min_level.upper(), 40)
        matches = [
            r for r in records
            if r.severity >= floor and (rule.contains is None or rule.contains.lower() in r.message.lower())
        ]
        out.append(
            {
                "rule": rule.name,
                "matches": len(matches),
                "threshold": rule.threshold,
                "fired": len(matches) >= rule.threshold,
                "sample": matches[0].message if matches else None,
            }
        )
    return out


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
