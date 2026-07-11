from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

DATA_CATEGORIES: Dict[str, str] = {
    "contact": "Contact details (name, email address, phone number)",
    "account": "Account credentials (username, hashed password)",
    "payment": "Payment information (billing address, last four card digits — processed by our payment provider)",
    "usage": "Usage data (pages visited, features used, session duration)",
    "device": "Device data (browser type, operating system, IP address)",
    "location": "Approximate location derived from IP address",
    "content": "Content you create or upload within the service",
}

GDPR_RIGHTS = [
    "Access — request a copy of the personal data we hold about you",
    "Rectification — correct inaccurate or incomplete data",
    "Erasure — ask us to delete your data ('right to be forgotten')",
    "Restriction — limit how we process your data",
    "Portability — receive your data in a machine-readable format",
    "Objection — object to processing based on legitimate interests",
    "Withdraw consent — at any time, without affecting prior processing",
]

CCPA_RIGHTS = [
    "Know — what personal information we collect, use, and disclose",
    "Delete — request deletion of your personal information",
    "Correct — request correction of inaccurate personal information",
    "Opt out — of the sale or sharing of your personal information",
    "Non-discrimination — equal service and price, even if you exercise your rights",
]


@dataclass
class CompanyProfile:
    company: str
    website: str
    contact_email: str
    data_categories: List[str] = field(default_factory=list)
    purposes: List[str] = field(default_factory=lambda: ["provide and improve the service"])
    uses_cookies: bool = True
    uses_analytics: bool = False
    third_parties: List[str] = field(default_factory=list)
    sells_data: bool = False
    serves_eu: bool = False
    serves_california: bool = False
    collects_children: bool = False
    retention_months: int = 24


def compliance_checklist(p: CompanyProfile) -> List[Dict[str, str]]:
    """Return [{level: ok|action|warning, item: str}] — what the policy covers and what still needs doing."""
    items: List[Dict[str, str]] = []
    if p.serves_eu:
        items.append({"level": "ok", "item": "GDPR rights section included (7 rights + supervisory authority)"})
        items.append({"level": "action", "item": "GDPR: document your lawful basis per purpose in a Records of Processing (Art. 30)"})
        if p.third_parties:
            items.append({"level": "action", "item": "GDPR: ensure Data Processing Agreements exist with each third-party processor"})
    if p.serves_california:
        items.append({"level": "ok", "item": "CCPA/CPRA rights section included (know, delete, correct, opt out)"})
        if p.sells_data:
            items.append({"level": "warning", "item": "CCPA: you sell/share data — a 'Do Not Sell or Share My Personal Information' link is REQUIRED on your homepage"})
        else:
            items.append({"level": "ok", "item": "CCPA: policy states you do not sell personal information"})
    if p.collects_children:
        items.append({"level": "warning", "item": "COPPA: collecting data from children under 13 requires verifiable parental consent — consult counsel before launch"})
    if p.uses_analytics:
        items.append({"level": "action", "item": "Cookie consent: analytics cookies need opt-in consent in the EU (ePrivacy) — add a consent banner"})
    if "payment" in p.data_categories:
        items.append({"level": "ok", "item": "Payment data handled by processor — policy states you never store full card numbers"})
    if not p.contact_email:
        items.append({"level": "warning", "item": "No contact email set — privacy laws require a working contact channel"})
    items.append({"level": "action", "item": "Review the generated policy with a qualified lawyer before publishing"})
    return items


def generate_policy(p: CompanyProfile, effective_date: str) -> str:
    s: List[str] = [f"# Privacy Policy — {p.company}", f"\n*Effective date: {effective_date}*\n"]
    s.append(f"This policy explains how {p.company} (\"we\", \"us\") collects, uses, and protects your personal "
             f"information when you use {p.website}.\n")

    s.append("## 1. Information We Collect\n")
    for cat in p.data_categories:
        s.append(f"- {DATA_CATEGORIES.get(cat, cat)}")
    if not p.data_categories:
        s.append("- We do not collect personal information beyond what you voluntarily provide.")

    s.append("\n## 2. How We Use Your Information\n")
    s.append("We use your information to:\n")
    for purpose in p.purposes:
        s.append(f"- {purpose[0].upper() + purpose[1:]}")
    if p.serves_eu:
        s.append("\nWhere GDPR applies, we rely on the following legal bases: performance of a contract "
                 "(providing the service), legitimate interests (improving and securing the service), and "
                 "consent (marketing and non-essential cookies).")

    s.append("\n## 3. Cookies\n")
    if p.uses_cookies:
        s.append("We use essential cookies to keep you signed in and remember preferences."
                 + (" We also use analytics cookies to understand how the service is used; "
                    "you can decline these via the cookie banner." if p.uses_analytics else ""))
    else:
        s.append("We do not use cookies.")

    s.append("\n## 4. Sharing Your Information\n")
    if p.third_parties:
        s.append("We share data only with service providers who help us operate the service:\n")
        for tp in p.third_parties:
            s.append(f"- {tp}")
        s.append("\nEach provider is bound by a data processing agreement and may not use your data for its own purposes.")
    else:
        s.append("We do not share your personal information with third parties.")
    s.append("\nWe **" + ("do" if p.sells_data else "do not") + "** sell your personal information."
             + (" California residents may opt out via the 'Do Not Sell or Share' link on our homepage."
                if p.sells_data and p.serves_california else ""))

    s.append(f"\n## 5. Data Retention\n\nWe keep personal data for up to **{p.retention_months} months** after your "
             "last activity, then delete or anonymize it — unless a longer period is required by law "
             "(e.g. tax and accounting records).")

    section = 6
    if p.serves_eu:
        s.append(f"\n## {section}. Your Rights (GDPR — EU/EEA/UK Residents)\n")
        s.extend(f"- **{r.split(' — ')[0]}** — {r.split(' — ')[1]}" for r in GDPR_RIGHTS)
        s.append(f"\nTo exercise any right, email {p.contact_email}. You may also lodge a complaint with your "
                 "local supervisory authority.")
        section += 1
    if p.serves_california:
        s.append(f"\n## {section}. Your Rights (CCPA/CPRA — California Residents)\n")
        s.extend(f"- **{r.split(' — ')[0]}** — {r.split(' — ')[1]}" for r in CCPA_RIGHTS)
        s.append(f"\nSubmit requests to {p.contact_email}. We verify identity before fulfilling requests and "
                 "respond within 45 days.")
        section += 1

    s.append(f"\n## {section}. Children's Privacy\n")
    if p.collects_children:
        s.append("Parts of the service are directed at children under 13. We obtain verifiable parental consent "
                 "before collecting personal information from children, per COPPA.")
    else:
        s.append("The service is not directed at children under 13, and we do not knowingly collect their data. "
                 "If you believe a child has provided us data, contact us and we will delete it.")
    section += 1

    s.append(f"\n## {section}. Security\n\nWe use industry-standard safeguards (encryption in transit, access "
             "controls, least-privilege) to protect your data. No method is 100% secure; we will notify you of "
             "any breach affecting your data as required by law.")
    section += 1

    s.append(f"\n## {section}. Changes to This Policy\n\nWe may update this policy and will post the new version "
             "here with a revised effective date. Material changes will be announced in the service.")
    section += 1

    s.append(f"\n## {section}. Contact\n\nQuestions or requests: **{p.contact_email}**")
    s.append("\n---\n*Generated as a starting draft — have a qualified lawyer review before publishing.*")
    return "\n".join(s)
