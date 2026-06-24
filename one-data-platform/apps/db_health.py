from __future__ import annotations

"""A mounted app: Database Health Dashboard.

This is the Day-10 portfolio build, now running *inside* the platform shell instead
of as a standalone Streamlit app. It follows the mount contract: a module that
exposes `render(ctx) -> str` returning an HTML fragment. The shell handles login,
the role check, the audit log, and the page chrome; the app just produces content.

It also demonstrates the connector layer (Step 4): it asks for the `demo_warehouse`
connection by name and runs a trivial query - proving an app gets data without ever
holding a credential.

Core scoring logic is vendored here (kept tiny + UI-free) so the platform is
self-contained.
"""

import random
from typing import Any, Dict, List

METRIC_SPECS: Dict[str, Dict[str, Any]] = {
    "cache_hit_ratio": {"label": "Cache hit ratio", "unit": "%", "dir": "low", "amber": 95.0, "red": 90.0},
    "conn_pool_usage": {"label": "Connection pool", "unit": "%", "dir": "high", "amber": 75.0, "red": 90.0},
    "avg_query_ms": {"label": "Avg query latency", "unit": "ms", "dir": "high", "amber": 100.0, "red": 250.0},
    "slow_queries": {"label": "Slow queries", "unit": "", "dir": "high", "amber": 5.0, "red": 15.0},
    "replication_lag_s": {"label": "Replication lag", "unit": "s", "dir": "high", "amber": 5.0, "red": 30.0},
    "disk_usage": {"label": "Disk usage", "unit": "%", "dir": "high", "amber": 75.0, "red": 90.0},
}
STATUS_POINTS = {"green": 100, "amber": 60, "red": 20}


def _simulate(seed: int = 7) -> Dict[str, float]:
    rng = random.Random(seed)
    j = lambda b, s: round(b + rng.uniform(-s, s), 1)
    return {"cache_hit_ratio": j(98, 1.5), "conn_pool_usage": j(48, 18), "avg_query_ms": j(60, 30),
            "slow_queries": float(rng.randint(0, 7)), "replication_lag_s": j(3, 3), "disk_usage": j(63, 14)}


def _score(name: str, value: float) -> str:
    s = METRIC_SPECS[name]
    a, r = float(s["amber"]), float(s["red"])
    if s["dir"] == "high":
        return "red" if value >= r else "amber" if value >= a else "green"
    return "red" if value <= r else "amber" if value <= a else "green"


def _report(metrics: Dict[str, float]) -> List[Dict[str, Any]]:
    rows = []
    for name, value in metrics.items():
        st = _score(name, value)
        rows.append({"label": METRIC_SPECS[name]["label"], "value": value,
                     "unit": METRIC_SPECS[name]["unit"], "status": st, "points": STATUS_POINTS[st]})
    return rows


def _connection_note(ctx: Dict[str, Any]) -> str:
    """Use the connector layer to prove data access without holding a credential."""
    get_connection = ctx.get("get_connection")
    if not get_connection:
        return ""
    try:
        conn = get_connection("demo_warehouse")
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS heartbeat (ok INTEGER)")
        cur.execute("INSERT INTO heartbeat VALUES (1)")
        n = cur.execute("SELECT COUNT(*) FROM heartbeat").fetchone()[0]
        conn.close()
        return (f"<p class='conn'>🔌 Connected to <code>demo_warehouse</code> via the connector "
                f"layer (no credential held by this app) - heartbeat rows: {n}.</p>")
    except Exception as exc:  # noqa: BLE001
        return f"<p class='conn'>connector note: {exc}</p>"


def render(ctx: Dict[str, Any]) -> str:
    """Mount contract: return an HTML fragment. `ctx` has the user + get_connection."""
    metrics = _simulate()
    rows = _report(metrics)
    score = round(sum(r["points"] for r in rows) / len(rows), 1)
    grade = "Healthy" if score >= 90 else "Watch" if score >= 70 else "Critical"
    dot = {"green": "🟢", "amber": "🟡", "red": "🔴"}

    cells = "".join(
        f"<tr><td>{r['label']}</td><td>{r['value']:g}{r['unit']}</td>"
        f"<td>{dot[r['status']]} {r['status']}</td></tr>" for r in rows
    )
    user = ctx.get("user", {})
    return f"""
<p>Live database health, rendered inside the shell for
<b>{user.get('email','?')}</b> (role <b>{user.get('role','?')}</b>).</p>
<div class="score">Health score: <b>{score}/100</b> · <b>{grade}</b></div>
{_connection_note(ctx)}
<table class="health"><tr><th>Metric</th><th>Value</th><th>Status</th></tr>{cells}</table>
<style>
.score{{font-size:1.1rem;margin:.6rem 0;padding:.6rem 1rem;background:#eef6f0;border-radius:8px;display:inline-block}}
.conn{{color:#3d34d6;font-size:.9rem}}
table.health{{border-collapse:collapse;margin-top:1rem;min-width:380px}}
table.health th,table.health td{{border:1px solid #e7e7f0;padding:.5rem .8rem;text-align:left}}
table.health th{{background:#f3f2fc;color:#3d34d6}}
</style>"""
