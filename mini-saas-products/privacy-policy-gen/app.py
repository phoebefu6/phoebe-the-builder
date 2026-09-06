from __future__ import annotations

import datetime

import streamlit as st
from policy_gen import DATA_CATEGORIES, CompanyProfile, compliance_checklist, generate_policy

st.set_page_config(page_title="Privacy Policy Generator", page_icon="🔒", layout="wide")
st.title("🔒 Privacy Policy Generator")
st.caption("Answer a short questionnaire → GDPR/CCPA-aware privacy policy draft + compliance checklist.")

with st.sidebar:
    st.header("Your company")
    company = st.text_input("Company name", "Acme Analytics")
    website = st.text_input("Website", "https://acme-analytics.example")
    contact = st.text_input("Privacy contact email", "privacy@acme-analytics.example")

col1, col2 = st.columns(2)
with col1:
    st.subheader("What you collect")
    cats = st.multiselect("Data categories", list(DATA_CATEGORIES),
                          default=["contact", "account", "usage", "device"],
                          format_func=lambda k: DATA_CATEGORIES[k].split(" (")[0])
    purposes = st.text_area("Purposes (one per line)",
                            "provide and improve the service\nrespond to support requests\nsend product updates you opt into").splitlines()
    retention = st.slider("Retention after last activity (months)", 1, 84, 24)
with col2:
    st.subheader("How you operate")
    uses_cookies = st.checkbox("Uses cookies", value=True)
    uses_analytics = st.checkbox("Uses analytics cookies (GA, Mixpanel...)", value=True)
    sells = st.checkbox("Sells or shares data for advertising", value=False)
    eu = st.checkbox("Serves EU/EEA/UK users (GDPR)", value=True)
    ca = st.checkbox("Serves California users (CCPA/CPRA)", value=True)
    children = st.checkbox("Directed at children under 13", value=False)
    third_parties = [t for t in st.text_area("Third-party processors (one per line)",
                                             "Stripe (payments)\nAWS (hosting)\nPostmark (transactional email)").splitlines() if t.strip()]

if st.button("Generate policy", type="primary"):
    profile = CompanyProfile(
        company=company, website=website, contact_email=contact,
        data_categories=cats, purposes=[p for p in purposes if p.strip()],
        uses_cookies=uses_cookies, uses_analytics=uses_analytics,
        third_parties=third_parties, sells_data=sells,
        serves_eu=eu, serves_california=ca, collects_children=children,
        retention_months=retention,
    )
    policy = generate_policy(profile, datetime.date.today().isoformat())
    checklist = compliance_checklist(profile)

    st.subheader("Compliance checklist")
    icons = {"ok": "✅", "action": "🔧", "warning": "⚠️"}
    for item in checklist:
        (st.warning if item["level"] == "warning" else st.info if item["level"] == "action" else st.success)(
            f"{icons[item['level']]} {item['item']}")

    st.markdown("---")
    st.subheader("Generated policy")
    st.markdown(policy)
    st.download_button("⬇️ Download policy (Markdown)", policy, file_name="privacy_policy.md", mime="text/markdown")
else:
    st.info("Fill in the questionnaire and click **Generate policy**.")
