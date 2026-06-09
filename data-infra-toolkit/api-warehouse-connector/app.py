"""API to Warehouse Connector — pull data from multiple SaaS APIs into a unified warehouse."""
from __future__ import annotations

import json
import hashlib
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path("warehouse.db")


# ---------------------------------------------------------------------------
# Mock API Sources — simulate real SaaS endpoints
# ---------------------------------------------------------------------------

def _mock_stripe_transactions() -> List[Dict[str, Any]]:
    import random, string
    random.seed(int(datetime.now().timestamp()) % 1000)
    rows = []
    for i in range(random.randint(15, 30)):
        rows.append({
            "transaction_id": "txn_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=12)),
            "amount_cents": random.randint(500, 50000),
            "currency": random.choice(["usd", "eur", "gbp"]),
            "status": random.choice(["succeeded", "succeeded", "succeeded", "failed", "pending"]),
            "created_at": (datetime.now() - timedelta(hours=random.randint(0, 72))).isoformat(),
        })
    return rows


def _mock_hubspot_contacts() -> List[Dict[str, Any]]:
    import random
    random.seed(42)
    first_names = ["Alex", "Jordan", "Sam", "Morgan", "Casey", "Riley", "Quinn", "Avery", "Taylor", "Drew"]
    companies = ["Acme Corp", "Globex", "Initech", "Umbrella", "Stark Industries", "Wayne Enterprises"]
    rows = []
    for i in range(random.randint(10, 20)):
        name = random.choice(first_names)
        rows.append({
            "contact_id": f"hs_{1000 + i}",
            "name": name,
            "email": f"{name.lower()}@{random.choice(companies).lower().replace(' ', '')}.com",
            "company": random.choice(companies),
            "lifecycle_stage": random.choice(["subscriber", "lead", "mql", "sql", "customer"]),
            "last_activity": (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
        })
    return rows


def _mock_github_events() -> List[Dict[str, Any]]:
    import random
    random.seed(99)
    repos = ["data-pipeline", "web-app", "ml-service", "infra-config", "docs"]
    actions = ["push", "pull_request.opened", "pull_request.merged", "issues.opened", "issues.closed", "release.published"]
    rows = []
    for i in range(random.randint(20, 40)):
        rows.append({
            "event_id": f"evt_{3000 + i}",
            "repo": random.choice(repos),
            "action": random.choice(actions),
            "actor": random.choice(["dev1", "dev2", "dev3", "bot"]),
            "created_at": (datetime.now() - timedelta(hours=random.randint(0, 168))).isoformat(),
        })
    return rows


def _mock_slack_messages() -> List[Dict[str, Any]]:
    import random
    random.seed(77)
    channels = ["#general", "#engineering", "#data-team", "#incidents", "#random"]
    rows = []
    for i in range(random.randint(25, 50)):
        rows.append({
            "message_id": f"msg_{5000 + i}",
            "channel": random.choice(channels),
            "user": random.choice(["alice", "bob", "carol", "dave", "eve"]),
            "word_count": random.randint(3, 120),
            "has_attachment": random.choice([True, False, False, False]),
            "timestamp": (datetime.now() - timedelta(hours=random.randint(0, 48))).isoformat(),
        })
    return rows


def _mock_weather_readings() -> List[Dict[str, Any]]:
    import random
    random.seed(55)
    cities = ["San Francisco", "New York", "Chicago", "Austin", "Seattle"]
    rows = []
    for i in range(random.randint(10, 15)):
        rows.append({
            "city": random.choice(cities),
            "temp_f": round(random.uniform(32, 105), 1),
            "humidity_pct": random.randint(20, 95),
            "condition": random.choice(["sunny", "cloudy", "rain", "fog", "snow"]),
            "recorded_at": (datetime.now() - timedelta(hours=random.randint(0, 24))).isoformat(),
        })
    return rows


CONNECTORS: Dict[str, Dict[str, Any]] = {
    "stripe": {"fetch": _mock_stripe_transactions, "table": "raw_stripe_transactions", "label": "Stripe Payments"},
    "hubspot": {"fetch": _mock_hubspot_contacts, "table": "raw_hubspot_contacts", "label": "HubSpot CRM"},
    "github": {"fetch": _mock_github_events, "table": "raw_github_events", "label": "GitHub Activity"},
    "slack": {"fetch": _mock_slack_messages, "table": "raw_slack_messages", "label": "Slack Messages"},
    "weather": {"fetch": _mock_weather_readings, "table": "raw_weather_readings", "label": "Weather Data"},
}


# ---------------------------------------------------------------------------
# Core ETL Engine
# ---------------------------------------------------------------------------

class WarehouseConnector:
    """Extract from SaaS APIs, transform, and load into SQLite warehouse."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self._ensure_meta_table()

    def _ensure_meta_table(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS _sync_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                table_name TEXT NOT NULL,
                rows_synced INTEGER NOT NULL,
                sync_hash TEXT,
                synced_at TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def extract(self, source: str) -> List[Dict[str, Any]]:
        cfg = CONNECTORS[source]
        logger.info("Extracting from %s...", cfg["label"])
        data = cfg["fetch"]()
        logger.info("  -> Got %d records", len(data))
        return data

    def transform(self, records: List[Dict[str, Any]], source: str) -> pd.DataFrame:
        df = pd.DataFrame(records)
        df["_source"] = source
        df["_loaded_at"] = datetime.now().isoformat()
        return df

    def load(self, df: pd.DataFrame, table: str, source: str) -> int:
        df.to_sql(table, self.conn, if_exists="replace", index=False)
        data_hash = hashlib.md5(df.to_json().encode()).hexdigest()[:12]
        self.conn.execute(
            "INSERT INTO _sync_log (source, table_name, rows_synced, sync_hash, synced_at) VALUES (?, ?, ?, ?, ?)",
            (source, table, len(df), data_hash, datetime.now().isoformat()),
        )
        self.conn.commit()
        logger.info("  -> Loaded %d rows into %s", len(df), table)
        return len(df)

    def sync(self, source: str) -> Dict[str, Any]:
        cfg = CONNECTORS[source]
        raw = self.extract(source)
        if not raw:
            return {"source": source, "status": "empty", "rows": 0}
        df = self.transform(raw, source)
        count = self.load(df, cfg["table"], source)
        return {"source": source, "status": "ok", "rows": count, "table": cfg["table"]}

    def sync_all(self) -> List[Dict[str, Any]]:
        return [self.sync(src) for src in CONNECTORS]

    def get_sync_history(self) -> pd.DataFrame:
        return pd.read_sql("SELECT * FROM _sync_log ORDER BY synced_at DESC", self.conn)

    def query(self, sql: str) -> pd.DataFrame:
        return pd.read_sql(sql, self.conn)

    def close(self) -> None:
        self.conn.close()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="API → Warehouse Connector", page_icon="🔌", layout="wide")
    st.title("🔌 API to Warehouse Connector")
    st.caption("Pull data from 5 SaaS sources into a local SQLite warehouse")

    wh = WarehouseConnector()

    tab1, tab2, tab3 = st.tabs(["Sync", "Explore", "History"])

    with tab1:
        st.subheader("Data Sources")
        cols = st.columns(len(CONNECTORS))
        for i, (key, cfg) in enumerate(CONNECTORS.items()):
            with cols[i]:
                if st.button(f"Sync {cfg['label']}", key=f"sync_{key}", use_container_width=True):
                    with st.spinner(f"Syncing {cfg['label']}..."):
                        result = wh.sync(key)
                    st.success(f"Loaded {result['rows']} rows into `{result['table']}`")

        st.divider()
        if st.button("Sync All Sources", type="primary", use_container_width=True):
            with st.spinner("Running full sync..."):
                results = wh.sync_all()
            for r in results:
                st.write(f"**{r['source']}** — {r['rows']} rows → `{r['table']}`")
            st.success("Full sync complete.")

    with tab2:
        st.subheader("Query Warehouse")
        tables = [cfg["table"] for cfg in CONNECTORS.values()]
        selected = st.selectbox("Table", tables)
        limit = st.slider("Rows", 5, 100, 20)
        try:
            df = wh.query(f"SELECT * FROM {selected} LIMIT {limit}")
            st.dataframe(df, use_container_width=True)
            st.caption(f"{len(df)} rows shown")
        except Exception:
            st.info("No data yet — run a sync first.")

    with tab3:
        st.subheader("Sync History")
        try:
            history = wh.get_sync_history()
            if history.empty:
                st.info("No syncs recorded yet.")
            else:
                st.dataframe(history, use_container_width=True)
        except Exception:
            st.info("No syncs recorded yet.")

    wh.close()


if __name__ == "__main__":
    main()
