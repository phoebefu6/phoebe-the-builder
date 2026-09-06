"""PII Detector and Masker — Streamlit app for scanning and redacting sensitive data."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# PII pattern definitions
# ---------------------------------------------------------------------------

class PIIType(Enum):
    EMAIL = "Email"
    PHONE_US = "Phone (US)"
    SSN = "SSN"
    CREDIT_CARD = "Credit Card"
    IP_ADDRESS = "IP Address"
    DATE_OF_BIRTH = "Date of Birth"
    US_ZIPCODE = "US Zip Code"


PII_PATTERNS: Dict[PIIType, re.Pattern] = {
    PIIType.EMAIL: re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    ),
    PIIType.PHONE_US: re.compile(
        r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"
    ),
    PIIType.SSN: re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    PIIType.CREDIT_CARD: re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    PIIType.IP_ADDRESS: re.compile(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    ),
    PIIType.DATE_OF_BIRTH: re.compile(
        r"\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b"
    ),
    PIIType.US_ZIPCODE: re.compile(r"\b\d{5}(?:-\d{4})?\b"),
}

MASK_CHAR = "█"


@dataclass
class PIIMatch:
    pii_type: PIIType
    value: str
    column: str
    row: int


@dataclass
class ScanResult:
    matches: List[PIIMatch] = field(default_factory=list)
    total_cells_scanned: int = 0
    columns_with_pii: Dict[str, Dict[str, int]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core detection logic
# ---------------------------------------------------------------------------

def mask_value(value: str, pii_type: PIIType) -> str:
    if pii_type == PIIType.EMAIL:
        local, domain = value.split("@", 1)
        return f"{local[0]}{MASK_CHAR * (len(local) - 1)}@{domain}"
    if pii_type == PIIType.SSN:
        return f"XXX-XX-{value[-4:]}"
    if pii_type == PIIType.CREDIT_CARD:
        digits = re.sub(r"\D", "", value)
        return f"{MASK_CHAR * (len(digits) - 4)}{digits[-4:]}"
    if pii_type == PIIType.PHONE_US:
        digits = re.sub(r"\D", "", value)
        if len(digits) >= 4:
            return f"{MASK_CHAR * (len(digits) - 4)}{digits[-4:]}"
    return MASK_CHAR * len(value)


def scan_dataframe(
    df: pd.DataFrame,
    selected_types: Optional[List[PIIType]] = None,
) -> ScanResult:
    if selected_types is None:
        selected_types = list(PIIType)

    result = ScanResult()
    patterns = {k: v for k, v in PII_PATTERNS.items() if k in selected_types}

    for col in df.columns:
        col_counts: Dict[str, int] = {}
        for row_idx, cell in enumerate(df[col].astype(str)):
            result.total_cells_scanned += 1
            for pii_type, pattern in patterns.items():
                for match in pattern.finditer(cell):
                    result.matches.append(
                        PIIMatch(pii_type, match.group(), col, row_idx)
                    )
                    col_counts[pii_type.value] = col_counts.get(pii_type.value, 0) + 1
        if col_counts:
            result.columns_with_pii[col] = col_counts
    return result


def mask_dataframe(
    df: pd.DataFrame,
    selected_types: Optional[List[PIIType]] = None,
) -> pd.DataFrame:
    if selected_types is None:
        selected_types = list(PIIType)

    masked = df.copy()
    patterns = {k: v for k, v in PII_PATTERNS.items() if k in selected_types}

    for col in masked.columns:
        for pii_type, pattern in patterns.items():
            masked[col] = masked[col].astype(str).apply(
                lambda cell, p=pattern, pt=pii_type: p.sub(
                    lambda m: mask_value(m.group(), pt), cell
                )
            )
    return masked


# ---------------------------------------------------------------------------
# Sample data for demo
# ---------------------------------------------------------------------------

def get_sample_data() -> pd.DataFrame:
    return pd.DataFrame({
        "name": ["Alice Johnson", "Bob Smith", "Carol Williams", "Dave Brown", "Eve Davis"],
        "email": [
            "alice.johnson@acme.com", "bob.smith@gmail.com",
            "carol.w@company.org", "dave.b@startup.io", "eve.davis@example.net",
        ],
        "phone": [
            "(555) 123-4567", "555-987-6543", "+1 555.246.8135",
            "555-369-2580", "(555) 147-2583",
        ],
        "ssn": [
            "123-45-6789", "987-65-4321", "456-78-9012",
            "321-54-9876", "654-32-1098",
        ],
        "notes": [
            "Contact at 192.168.1.100", "DOB: 03/15/1990",
            "Card: 4111-1111-1111-1111", "Zip: 94102-3456",
            "Alt email: eve2@backup.com",
        ],
    })


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="PII Detector & Masker", page_icon="🔒", layout="wide")
    st.title("🔒 PII Detector & Masker")
    st.caption("Scan CSV data for personally identifiable information and mask it before sharing.")

    st.sidebar.header("Settings")
    selected = st.sidebar.multiselect(
        "PII types to detect",
        options=[t.value for t in PIIType],
        default=[t.value for t in PIIType],
    )
    selected_types = [t for t in PIIType if t.value in selected]

    data_source = st.radio("Data source", ["Sample data", "Upload CSV"], horizontal=True)

    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV to get started, or switch to sample data.")
            return
        df = pd.read_csv(uploaded)
    else:
        df = get_sample_data()

    st.subheader("Original Data")
    st.dataframe(df, use_container_width=True)

    if st.button("🔍 Scan for PII", type="primary"):
        result = scan_dataframe(df, selected_types)

        col1, col2, col3 = st.columns(3)
        col1.metric("PII Matches Found", len(result.matches))
        col2.metric("Cells Scanned", result.total_cells_scanned)
        col3.metric("Columns with PII", len(result.columns_with_pii))

        if result.matches:
            st.subheader("PII Findings")
            findings_df = pd.DataFrame([
                {"Type": m.pii_type.value, "Value": m.value, "Column": m.column, "Row": m.row}
                for m in result.matches
            ])
            st.dataframe(findings_df, use_container_width=True)

            st.subheader("Column Risk Summary")
            for col_name, counts in result.columns_with_pii.items():
                with st.expander(f"**{col_name}** — {sum(counts.values())} PII items"):
                    for pii_name, count in counts.items():
                        st.write(f"- {pii_name}: {count}")

            st.subheader("Masked Data")
            masked_df = mask_dataframe(df, selected_types)
            st.dataframe(masked_df, use_container_width=True)

            csv_buf = io.StringIO()
            masked_df.to_csv(csv_buf, index=False)
            st.download_button(
                "⬇️ Download Masked CSV",
                csv_buf.getvalue(),
                file_name="masked_data.csv",
                mime="text/csv",
            )
        else:
            st.success("No PII detected in the dataset!")


if __name__ == "__main__":
    main()
