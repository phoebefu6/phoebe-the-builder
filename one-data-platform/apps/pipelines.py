from __future__ import annotations

"""A mounted app: Pipelines - govern scheduled work.

Surfaces the orchestration layer (Step 6) inside the shell: every DAG, its schedule,
owner, and last-run status. The backend is Apache Airflow when configured, or a local
simulation otherwise - this app only uses the orchestrator interface, so it doesn't
care which. Follows the mount contract: `render(ctx) -> html`.
"""

import sys
from pathlib import Path
from typing import Any, Dict

# The orchestration layer lives in ../../orchestration relative to the platform root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "orchestration"))
from orchestrator import get_orchestrator  # noqa: E402


def render(ctx: Dict[str, Any]) -> str:
    orch = get_orchestrator()
    dags = orch.list_dags()
    dot = {"success": "🟢", "failed": "🔴", "queued": "🟡", "unknown": "⚪"}

    rows = "".join(
        f"<tr><td><b>{d['id']}</b><br><span class='desc'>{d['description']}</span></td>"
        f"<td><code>{d['schedule']}</code></td><td>{d['owner']}</td>"
        f"<td>{dot.get(d['last_status'],'⚪')} {d['last_status']}</td></tr>"
        for d in dags
    )
    failed = sum(1 for d in dags if d["last_status"] == "failed")
    return f"""
<p>Scheduled pipelines, governed under the shell. Backend:
<b>{orch.backend}</b> {'(set AIRFLOW_URL to use a real Airflow)' if orch.backend=='local-sim' else ''}.</p>
<div class="pstat">{len(dags)} pipeline(s) · {failed} failing</div>
<table class="pipes"><tr><th>Pipeline</th><th>Schedule (cron)</th><th>Owner</th><th>Last run</th></tr>{rows}</table>
<p class="note">We don't reimplement scheduling - we plug in <b>Apache Airflow</b> (open source)
and govern access to it. Triggering a run is a governed, audited action via the
orchestrator interface.</p>
<style>
.pstat{{font-size:1.05rem;margin:.6rem 0;padding:.5rem .9rem;background:#eef6f0;border-radius:8px;display:inline-block}}
table.pipes{{border-collapse:collapse;margin-top:1rem;width:100%}}
table.pipes th,table.pipes td{{border:1px solid #e7e7f0;padding:.55rem .8rem;text-align:left;vertical-align:top}}
table.pipes th{{background:#f3f2fc;color:#3d34d6}}
.desc{{color:#777;font-size:.85rem}} .note{{color:#555;font-size:.9rem;margin-top:1rem}}
</style>"""
