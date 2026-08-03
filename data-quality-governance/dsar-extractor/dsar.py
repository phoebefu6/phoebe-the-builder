"""DSAR extractor: resolve a data subject across a warehouse, disclose only what is theirs.

Four mechanisms, in the order they matter:

1. IDENTITY RESOLUTION - one person is many keys (case variants, plus-addressing,
   pre-login anonymous ids). An exact-match WHERE clause under-collects.
2. ABSTENTION - weak links (same name, same household) are NOT auto-included.
   Over-resolving discloses a different person's data to the requester, which is
   the same breach in the opposite direction.
3. SCOPE CLASSIFICATION - following every foreign key outward pulls in the product
   catalog, which is not personal data about anyone. Each edge is declared as
   subject-scoped or reference.
4. THIRD-PARTY WITHHOLDING - shared rows (a ticket thread, a referral, a gift order)
   contain another living person. Art. 15(4) / PDPA s.21(3): the copy must not
   adversely affect the rights of others. Ship it raw and the disclosure is a breach.

Plus: access is not erasure. Rows you must disclose under Art. 15 may be rows you
cannot delete under Art. 17, because a statutory retention period outranks the
erasure right. The plan says which, and on what basis.

No database, no network. Corpus is generated deterministically.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence, Set, Tuple

Row = Dict[str, object]
Corpus = Dict[str, List[Row]]

TODAY = date(2026, 8, 3)


# --------------------------------------------------------------------------------------
# 1. Identity
# --------------------------------------------------------------------------------------

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def normalize_email(value: str) -> str:
    """Case-fold and strip plus-addressing. Both are the same mailbox at every major provider.

    Deliberately does NOT strip dots: that is a Gmail-only convention, and applying it to
    a corporate domain merges two different people. Under-normalizing loses rows; this
    particular over-normalization discloses someone else's.
    """
    value = value.strip().lower()
    if "@" not in value:
        return value
    local, _, domain = value.partition("@")
    local = local.split("+", 1)[0]
    return f"{local}@{domain}"


@dataclass(frozen=True)
class ResolvedKey:
    """One identifier believed to belong to the subject, with why we believe it."""

    key_type: str  # "email" | "customer_id" | "anon_id" | "phone"
    value: str
    strength: str  # "seed" | "exact" | "linked" | "weak"
    evidence: str

    @property
    def usable(self) -> bool:
        """Weak links are surfaced for a human, never joined on automatically."""
        return self.strength in ("seed", "exact", "linked")


@dataclass
class Identity:
    keys: List[ResolvedKey] = field(default_factory=list)
    # Raw stored spellings that normalized onto a subject mailbox. Not extra keys - the
    # same key written differently - but the audit log has to show them, because they are
    # exactly what the naive WHERE clause failed to match.
    variants: List[str] = field(default_factory=list)

    def add(self, key: ResolvedKey) -> bool:
        for existing in self.keys:
            if existing.key_type == key.key_type and existing.value == key.value:
                return False
        self.keys.append(key)
        return True

    def values(self, key_type: str, include_weak: bool = False) -> Set[str]:
        return {
            k.value
            for k in self.keys
            if k.key_type == key_type and (include_weak or k.usable)
        }

    @property
    def weak(self) -> List[ResolvedKey]:
        return [k for k in self.keys if not k.usable]


def resolve_identity(corpus: Corpus, seed_email: str) -> Identity:
    """Expand one seed identifier into every key the subject is known by.

    A real DSAR arrives with an email address and nothing else. The subject exists in
    the warehouse under at least four different key shapes.
    """
    ident = Identity()
    seed = normalize_email(seed_email)
    ident.add(ResolvedKey("email", seed, "seed", "supplied on the request form"))

    # Pass 1: every stored email that normalizes to a known mailbox is the same mailbox.
    for table, column in EMAIL_COLUMNS:
        for row in corpus[table]:
            raw = str(row.get(column) or "")
            if not raw:
                continue
            norm = normalize_email(raw)
            if norm not in ident.values("email"):
                continue
            ident.add(
                ResolvedKey(
                    "email", norm, "exact", f"{table}.{column} normalizes to the seed mailbox"
                )
            )
            if raw.strip() != norm:
                note = f"{table}.{column} = {raw!r} -> {norm}"
                if note not in ident.variants:
                    ident.variants.append(note)

    # Pass 2: customer records reachable from those mailboxes.
    for row in corpus["customers"]:
        if normalize_email(str(row["email"])) in ident.values("email"):
            ident.add(
                ResolvedKey(
                    "customer_id",
                    str(row["customer_id"]),
                    "exact",
                    f"customers.email = {row['email']!r}",
                )
            )
            if row.get("phone"):
                ident.add(
                    ResolvedKey(
                        "phone",
                        str(row["phone"]),
                        "exact",
                        f"customers.phone on {row['customer_id']}",
                    )
                )

    # Pass 3: controller-asserted device links. Deterministic, so joinable - but they are
    # an assertion the controller made, not a fact, which is why they are tagged 'linked'.
    for row in corpus["identity_links"]:
        if str(row["customer_id"]) in ident.values("customer_id"):
            ident.add(
                ResolvedKey(
                    "anon_id",
                    str(row["anon_id"]),
                    "linked",
                    f"identity_links: device stitched on {row['linked_at']} ({row['method']})",
                )
            )

    # Pass 4: weak candidates. Recorded, never joined on. Including these is how a
    # controller mails one person's order history to a stranger with the same name.
    subject_names = {
        str(r["full_name"]).lower()
        for r in corpus["customers"]
        if str(r["customer_id"]) in ident.values("customer_id")
    }
    for row in corpus["customers"]:
        if str(row["customer_id"]) in ident.values("customer_id"):
            continue
        if str(row["full_name"]).lower() in subject_names:
            ident.add(
                ResolvedKey(
                    "customer_id",
                    str(row["customer_id"]),
                    "weak",
                    f"same full_name {row['full_name']!r}, different mailbox "
                    f"({row['email']}) - needs a human decision",
                )
            )
    return ident


# --------------------------------------------------------------------------------------
# 2. The subject map
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class KeyCol:
    column: str
    key_type: str
    role: str = "subject"  # "subject" identifies the row's owner; "counterparty" the other person


@dataclass(frozen=True)
class ScopedVia:
    """This table inherits subject scope from a parent row."""

    child_col: str
    parent_table: str
    parent_col: str


@dataclass(frozen=True)
class Retention:
    basis: str
    years: int
    date_col: str


@dataclass(frozen=True)
class TableSpec:
    name: str
    pk: str
    keys: Tuple[KeyCol, ...] = ()
    text_cols: Tuple[str, ...] = ()
    scoped_via: Optional[ScopedVia] = None
    category: str = "personal"  # "personal" | "reference"
    erasure: str = "delete"  # "delete" | "anonymize" | "redact"
    retention: Optional[Retention] = None
    fks: Tuple[Tuple[str, str, str], ...] = ()  # (col, target_table, target_col) - all of them


SUBJECT_MAP: Tuple[TableSpec, ...] = (
    TableSpec(
        "customers",
        "customer_id",
        keys=(
            KeyCol("email", "email"),
            KeyCol("customer_id", "customer_id"),
            KeyCol("phone", "phone"),
        ),
    ),
    TableSpec(
        "orders",
        "order_id",
        keys=(
            KeyCol("customer_id", "customer_id"),
            KeyCol("gift_recipient_email", "email", role="counterparty"),
        ),
        fks=(("customer_id", "customers", "customer_id"),),
        retention=Retention("Statutory: accounting records, 7 years", 7, "order_date"),
    ),
    TableSpec(
        "order_items",
        "order_item_id",
        scoped_via=ScopedVia("order_id", "orders", "order_id"),
        fks=(("order_id", "orders", "order_id"), ("product_id", "products", "product_id")),
        retention=Retention("Statutory: accounting records, 7 years", 7, "_order_date"),
    ),
    TableSpec(
        "payments",
        "payment_id",
        scoped_via=ScopedVia("order_id", "orders", "order_id"),
        fks=(("order_id", "orders", "order_id"),),
        retention=Retention("Statutory: tax and payment records, 7 years", 7, "paid_at"),
    ),
    TableSpec("products", "product_id", category="reference"),
    TableSpec(
        "support_tickets",
        "ticket_id",
        keys=(KeyCol("requester_email", "email"),),
        text_cols=("subject", "body"),
    ),
    TableSpec(
        "ticket_messages",
        "message_id",
        keys=(KeyCol("author_email", "email"),),
        text_cols=("body",),
        scoped_via=ScopedVia("ticket_id", "support_tickets", "ticket_id"),
        fks=(("ticket_id", "support_tickets", "ticket_id"),),
        erasure="redact",
    ),
    TableSpec(
        "marketing_contacts",
        "contact_id",
        keys=(KeyCol("email_lower", "email"),),
    ),
    TableSpec(
        "web_events",
        "event_id",
        keys=(KeyCol("customer_id", "customer_id"), KeyCol("anon_id", "anon_id")),
        fks=(("customer_id", "customers", "customer_id"),),
        erasure="anonymize",
    ),
    TableSpec(
        "identity_links",
        "link_id",
        keys=(KeyCol("customer_id", "customer_id"), KeyCol("anon_id", "anon_id")),
        fks=(("customer_id", "customers", "customer_id"),),
    ),
    TableSpec(
        "referrals",
        "referral_id",
        keys=(
            KeyCol("referrer_customer_id", "customer_id"),
            KeyCol("referee_email", "email"),
        ),
        erasure="redact",
    ),
)

SPEC_BY_NAME = {t.name: t for t in SUBJECT_MAP}

EMAIL_COLUMNS: Tuple[Tuple[str, str], ...] = tuple(
    (t.name, k.column) for t in SUBJECT_MAP for k in t.keys if k.key_type == "email"
)


# --------------------------------------------------------------------------------------
# 3. Extraction
# --------------------------------------------------------------------------------------

DIRECT = "direct"  # row carries a subject key
DERIVED = "derived"  # reached through a declared subject-scoped edge
MENTION = "mention"  # only free text names the subject; no key join reaches it


@dataclass
class Hit:
    table: str
    pk: str
    row: Row
    how: str  # DIRECT | DERIVED | MENTION
    reason: str
    third_parties: Tuple[str, ...] = ()

    @property
    def shared(self) -> bool:
        return bool(self.third_parties)


def _row_matches_subject(spec: TableSpec, row: Row, ident: Identity) -> Optional[str]:
    for key in spec.keys:
        if key.role != "subject":
            continue
        raw = row.get(key.column)
        if raw in (None, ""):
            continue
        value = normalize_email(str(raw)) if key.key_type == "email" else str(raw)
        if value in ident.values(key.key_type):
            return f"{spec.name}.{key.column} = {raw!r}"
    return None


def _third_parties(
    spec: TableSpec, row: Row, ident: Identity, others: Dict[str, str]
) -> Tuple[str, ...]:
    """Identifiers in this row that belong to a living person who is not the subject."""
    found: List[str] = []
    subject_emails = ident.values("email")
    subject_customers = ident.values("customer_id")

    for key in spec.keys:
        raw = row.get(key.column)
        if raw in (None, ""):
            continue
        if key.key_type == "email":
            norm = normalize_email(str(raw))
            if norm not in subject_emails and norm in others:
                found.append(f"{key.column}={raw}")
        elif key.key_type == "customer_id" and str(raw) not in subject_customers:
            found.append(f"{key.column}={raw}")

    for col in spec.text_cols:
        for match in EMAIL_RE.findall(str(row.get(col) or "")):
            norm = normalize_email(match)
            if norm not in subject_emails and norm in others:
                found.append(f"{col}~{match}")
    return tuple(dict.fromkeys(found))


def _mentions_subject(spec: TableSpec, row: Row, ident: Identity) -> Optional[str]:
    subject_emails = ident.values("email")
    for col in spec.text_cols:
        text = str(row.get(col) or "")
        for match in EMAIL_RE.findall(text):
            if normalize_email(match) in subject_emails:
                return f"free text {spec.name}.{col} contains {match!r}"
    return None


def _known_emails(corpus: Corpus) -> Dict[str, str]:
    """Every mailbox the controller holds a person behind, normalized -> display form."""
    known: Dict[str, str] = {}
    for row in corpus["customers"]:
        known[normalize_email(str(row["email"]))] = str(row["full_name"])
    for row in corpus["marketing_contacts"]:
        known.setdefault(normalize_email(str(row["email_lower"])), "marketing contact")
    return known


def extract(corpus: Corpus, ident: Identity) -> List[Hit]:
    """Everything the controller holds about this subject, and nothing only related to them."""
    others = _known_emails(corpus)
    hits: List[Hit] = []
    seen: Set[Tuple[str, str]] = set()

    def push(spec: TableSpec, row: Row, how: str, reason: str) -> None:
        pk = str(row[spec.pk])
        if (spec.name, pk) in seen:
            return
        seen.add((spec.name, pk))
        hits.append(Hit(spec.name, pk, row, how, reason, _third_parties(spec, row, ident, others)))

    # Pass A - direct key matches.
    for spec in SUBJECT_MAP:
        if spec.category == "reference":
            continue
        for row in corpus[spec.name]:
            reason = _row_matches_subject(spec, row, ident)
            if reason:
                push(spec, row, DIRECT, reason)

    # Pass B - inherit scope along declared subject-scoped edges, to a fixed point.
    for _ in range(len(SUBJECT_MAP)):
        added = False
        parents = {(h.table, h.pk) for h in hits}
        for spec in SUBJECT_MAP:
            if spec.scoped_via is None or spec.category == "reference":
                continue
            via = spec.scoped_via
            for row in corpus[spec.name]:
                if (spec.name, str(row[spec.pk])) in seen:
                    continue
                if (via.parent_table, str(row[via.child_col])) in parents:
                    before = len(hits)
                    push(
                        spec,
                        row,
                        DERIVED,
                        f"subject-scoped edge {spec.name}.{via.child_col} -> "
                        f"{via.parent_table}.{via.parent_col}",
                    )
                    added = added or len(hits) > before
        if not added:
            break

    # Pass C - free-text sweep. Rows about the subject that no key join reaches.
    for spec in SUBJECT_MAP:
        if spec.category == "reference" or not spec.text_cols:
            continue
        for row in corpus[spec.name]:
            if (spec.name, str(row[spec.pk])) in seen:
                continue
            reason = _mentions_subject(spec, row, ident)
            if reason:
                push(spec, row, MENTION, reason)

    return hits


# --------------------------------------------------------------------------------------
# 4. Baselines to measure against
# --------------------------------------------------------------------------------------


def naive_extract(corpus: Corpus, seed_email: str) -> Set[Tuple[str, str]]:
    """A competent-but-ordinary DSAR query: case-insensitive email match, then follow
    customer_id. This is what most controllers actually run."""
    target = seed_email.strip().lower()
    found: Set[Tuple[str, str]] = set()
    customer_ids: Set[str] = set()

    for table, column in EMAIL_COLUMNS:
        spec = SPEC_BY_NAME[table]
        for row in corpus[table]:
            if str(row.get(column) or "").strip().lower() == target:
                found.add((table, str(row[spec.pk])))
                if table == "customers":
                    customer_ids.add(str(row["customer_id"]))

    for spec in SUBJECT_MAP:
        if spec.category == "reference":
            continue
        if not any(k.column == "customer_id" for k in spec.keys):
            continue
        for row in corpus[spec.name]:
            if str(row.get("customer_id") or "") in customer_ids:
                found.add((spec.name, str(row[spec.pk])))

    # ... and the child tables of those orders, which the naive version does get right.
    order_ids = {pk for (t, pk) in found if t == "orders"}
    for name in ("order_items", "payments"):
        spec = SPEC_BY_NAME[name]
        for row in corpus[name]:
            if str(row["order_id"]) in order_ids:
                found.add((name, str(row[spec.pk])))
    return found


def naive_fk_sweep(corpus: Corpus, hits: Sequence[Hit]) -> Dict[str, int]:
    """Follow every foreign key outward from the included rows and count what arrives.

    This is the other common implementation: 'just traverse the schema graph'. It reaches
    the product catalog, which is not personal data about anybody.
    """
    index = {spec.name: {str(r[spec.pk]): r for r in corpus[spec.name]} for spec in SUBJECT_MAP}
    frontier = [(h.table, h.pk) for h in hits]
    seen = set(frontier)
    extra: Dict[str, Set[str]] = {}

    for _ in range(3):
        nxt: List[Tuple[str, str]] = []
        for table, pk in frontier:
            spec = SPEC_BY_NAME[table]
            row = index[table].get(pk)
            if row is None:
                continue
            for col, target, _target_col in spec.fks:
                value = row.get(col)
                if value in (None, ""):
                    continue
                tid = str(value)
                if (target, tid) in seen or tid not in index[target]:
                    continue
                seen.add((target, tid))
                nxt.append((target, tid))
                if SPEC_BY_NAME[target].category == "reference":
                    extra.setdefault(target, set()).add(tid)
        if not nxt:
            break
        frontier = nxt

    result = {t: len(v) for t, v in extra.items()}

    # The reverse join, for scale: one careless products -> order_items step.
    if "products" in extra:
        touched = [r for r in corpus["order_items"] if str(r["product_id"]) in extra["products"]]
        order_owner = {str(o["order_id"]): str(o["customer_id"]) for o in corpus["orders"]}
        result["_reverse_order_items"] = len(touched)
        result["_reverse_customers"] = len(
            {order_owner.get(str(r["order_id"])) for r in touched} - {None}
        )
    return result


def weak_link_cost(corpus: Corpus, ident: Identity) -> Dict[str, int]:
    """What auto-including the weak (same-name) links would have disclosed."""
    weak_ids = {k.value for k in ident.weak if k.key_type == "customer_id"}
    if not weak_ids:
        return {}
    orders = [r for r in corpus["orders"] if str(r["customer_id"]) in weak_ids]
    order_ids = {str(r["order_id"]) for r in orders}
    return {
        "other_people": len(weak_ids),
        "orders": len(orders),
        "payments": len([r for r in corpus["payments"] if str(r["order_id"]) in order_ids]),
        "order_items": len([r for r in corpus["order_items"] if str(r["order_id"]) in order_ids]),
    }


# --------------------------------------------------------------------------------------
# 5. Redaction and the disclosure pack
# --------------------------------------------------------------------------------------

WITHHELD = "[third party - withheld under Art. 15(4)]"


def redact(hit: Hit, ident: Identity, corpus: Corpus) -> Row:
    """Return a copy of the row safe to hand to the requester."""
    if not hit.shared:
        return dict(hit.row)

    spec = SPEC_BY_NAME[hit.table]
    others = _known_emails(corpus)
    out = dict(hit.row)
    subject_emails = ident.values("email")

    for key in spec.keys:
        raw = out.get(key.column)
        if raw in (None, ""):
            continue
        if key.key_type == "email":
            norm = normalize_email(str(raw))
            if norm not in subject_emails and norm in others:
                out[key.column] = WITHHELD
        elif key.key_type == "customer_id" and str(raw) not in ident.values("customer_id"):
            out[key.column] = WITHHELD

    for col in spec.text_cols:
        text = str(out.get(col) or "")
        if not text:
            continue
        for match in EMAIL_RE.findall(text):
            norm = normalize_email(match)
            if norm not in subject_emails and norm in others:
                text = text.replace(match, WITHHELD)
        out[col] = text
    return out


def disclosure_pack(corpus: Corpus, ident: Identity, hits: Sequence[Hit]) -> Dict[str, List[Row]]:
    pack: Dict[str, List[Row]] = {}
    for hit in hits:
        pack.setdefault(hit.table, []).append(redact(hit, ident, corpus))
    return pack


# --------------------------------------------------------------------------------------
# 6. Erasure plan - deliberately not the same set of rows
# --------------------------------------------------------------------------------------

DELETE = "delete"
ANONYMIZE = "anonymize"
REDACT_SUBJECT = "redact subject fields, keep row"
RETAIN = "retain - erasure blocked"


@dataclass
class ErasureAction:
    table: str
    pk: str
    action: str
    basis: str


def erasure_plan(hits: Sequence[Hit], corpus: Corpus, today: date = TODAY) -> List[ErasureAction]:
    order_date = {str(o["order_id"]): o["order_date"] for o in corpus["orders"]}
    plan: List[ErasureAction] = []

    for hit in hits:
        spec = SPEC_BY_NAME[hit.table]

        if spec.retention is not None:
            col = spec.retention.date_col
            when = (
                order_date.get(str(hit.row.get("order_id")))
                if col.startswith("_")
                else hit.row.get(col)
            )
            if isinstance(when, date) and when > today - timedelta(days=365 * spec.retention.years):
                until = when + timedelta(days=365 * spec.retention.years)
                plan.append(
                    ErasureAction(
                        hit.table,
                        hit.pk,
                        RETAIN,
                        f"{spec.retention.basis} - releases {until.isoformat()}",
                    )
                )
                continue

        if hit.shared:
            plan.append(
                ErasureAction(
                    hit.table,
                    hit.pk,
                    REDACT_SUBJECT,
                    "row is also another person's record - deleting it would erase their data",
                )
            )
        elif spec.erasure == ANONYMIZE:
            plan.append(
                ErasureAction(
                    hit.table,
                    hit.pk,
                    ANONYMIZE,
                    "null the identifier, keep the event for aggregates",
                )
            )
        elif hit.how == MENTION:
            plan.append(
                ErasureAction(
                    hit.table,
                    hit.pk,
                    REDACT_SUBJECT,
                    "subject appears only inside free text written by someone else",
                )
            )
        else:
            plan.append(ErasureAction(hit.table, hit.pk, DELETE, "no retention obligation"))
    return plan


# --------------------------------------------------------------------------------------
# 7. Deterministic corpus
# --------------------------------------------------------------------------------------

FIRST = ["Amara", "Ravi", "Mei", "Tomas", "Nadia", "Jonas", "Priya", "Ines", "Kofi", "Yuki"]
LAST = ["Osei", "Bakshi", "Lim", "Novak", "Haddad", "Berg", "Rao", "Costa", "Mensah", "Sato"]
CATEGORIES = ["kitchen", "outdoor", "audio", "lighting", "storage"]
SUBJECT_EMAIL = "amara.osei@example.com"


def build_corpus(seed: int = 7) -> Corpus:
    rng = random.Random(seed)
    customers: List[Row] = []

    # The subject. Stored with a display-cased mailbox - a case-sensitive WHERE misses it.
    customers.append(
        {
            "customer_id": "C0001",
            "full_name": "Amara Osei",
            "email": "Amara.Osei@Example.com",
            "phone": "+65 8100 4417",
            "created_at": date(2023, 3, 14),
        }
    )
    # A different living person with the same name. The reason weak links abstain.
    customers.append(
        {
            "customer_id": "C0047",
            "full_name": "Amara Osei",
            "email": "a.osei@other-example.com",
            "phone": "+65 9022 7781",
            "created_at": date(2024, 11, 2),
        }
    )
    # The three counterparties the subject is entangled with. Real customers, so the
    # third-party test has an actual person to protect rather than an unknown address.
    for cid, name, email in (
        ("C0012", "Jonas Berg", "jonas.berg12@example.com"),
        ("C0008", "Priya Rao", "priya.rao8@example.com"),
        ("C0021", "Yuki Sato", "yuki.sato21@example.com"),
    ):
        customers.append(
            {
                "customer_id": cid,
                "full_name": name,
                "email": email,
                "phone": f"+65 9{rng.randint(1000000, 9999999)}",
                "created_at": date(2023, 1, 1) + timedelta(days=rng.randint(0, 900)),
            }
        )

    taken = {str(c["customer_id"]) for c in customers}
    for i in range(2, 60):
        cid = f"C{i + 1:04d}"
        if cid in taken:
            continue
        first = rng.choice(FIRST)
        last = rng.choice(LAST)
        customers.append(
            {
                "customer_id": cid,
                "full_name": f"{first} {last}",
                "email": f"{first.lower()}.{last.lower()}{i}@example.com",
                "phone": f"+65 9{rng.randint(1000000, 9999999)}",
                "created_at": date(2023, 1, 1) + timedelta(days=rng.randint(0, 900)),
            }
        )

    products = [
        {
            "product_id": f"P{i:03d}",
            "name": f"{rng.choice(CATEGORIES).title()} item {i}",
            "category": rng.choice(CATEGORIES),
            "list_price": round(rng.uniform(8, 240), 2),
        }
        for i in range(1, 61)
    ]

    orders: List[Row] = []
    order_items: List[Row] = []
    payments: List[Row] = []
    oid = 0
    pmt = 0
    iid = 0

    def add_order(customer_id: str, when: date, gift: Optional[str] = None) -> str:
        nonlocal oid, pmt, iid
        oid += 1
        order_id = f"O{oid:05d}"
        total = 0.0
        for _ in range(rng.randint(1, 4)):
            iid += 1
            product = rng.choice(products)
            qty = rng.randint(1, 3)
            line = round(float(product["list_price"]) * qty, 2)
            total += line
            order_items.append(
                {
                    "order_item_id": f"OI{iid:05d}",
                    "order_id": order_id,
                    "product_id": product["product_id"],
                    "qty": qty,
                    "line_total": line,
                }
            )
        orders.append(
            {
                "order_id": order_id,
                "customer_id": customer_id,
                "order_date": when,
                "total": round(total, 2),
                "status": rng.choice(["fulfilled", "fulfilled", "refunded"]),
                "gift_recipient_email": gift,
            }
        )
        pmt += 1
        payments.append(
            {
                "payment_id": f"PM{pmt:05d}",
                "order_id": order_id,
                "amount": round(total, 2),
                "method": rng.choice(["card", "card", "paynow"]),
                "card_last4": f"{rng.randint(1000, 9999)}",
                "paid_at": when,
            }
        )
        return order_id

    # Subject's orders: some old enough to be erasable, some inside the 7-year hold.
    for when in [
        date(2018, 5, 9),
        date(2019, 2, 21),
        date(2023, 6, 30),
        date(2024, 4, 12),
        date(2025, 9, 3),
    ]:
        add_order("C0001", when)
    add_order("C0001", date(2025, 12, 20), gift="jonas.berg12@example.com")

    for customer in customers[1:]:
        for _ in range(rng.randint(1, 5)):
            add_order(
                str(customer["customer_id"]),
                date(2019, 1, 1) + timedelta(days=rng.randint(0, 2600)),
            )

    # Support: the subject writes in from a plus-addressed mailbox.
    support_tickets: List[Row] = [
        {
            "ticket_id": "T0001",
            "requester_email": "amara.osei+support@example.com",
            "subject": "Order O00003 arrived damaged",
            "body": "The lid was cracked on arrival. Photos attached.",
            "created_at": date(2023, 7, 2),
        },
        {
            "ticket_id": "T0002",
            "requester_email": "amara.osei+shop@example.com",
            "subject": "Change delivery address",
            "body": "Please redirect to the office address on file.",
            "created_at": date(2024, 4, 15),
        },
        {
            "ticket_id": "T0003",
            "requester_email": "Amara.Osei@Example.com",
            "subject": "Marketing emails",
            "body": "I would like to stop receiving the weekly newsletter.",
            "created_at": date(2025, 9, 10),
        },
    ]
    for i, customer in enumerate(customers[2:22], start=4):
        support_tickets.append(
            {
                "ticket_id": f"T{i:04d}",
                "requester_email": str(customer["email"]),
                "subject": f"Question about order {rng.randint(1, 90)}",
                "body": "Following up on my last message.",
                "created_at": date(2024, 1, 1) + timedelta(days=rng.randint(0, 600)),
            }
        )

    ticket_messages: List[Row] = [
        {
            "message_id": "M0001",
            "ticket_id": "T0001",
            "author_email": "amara.osei+support@example.com",
            "body": "Attaching two photos of the crack.",
            "sent_at": date(2023, 7, 2),
        },
        {
            "message_id": "M0002",
            "ticket_id": "T0001",
            "author_email": "support@shop.example",
            "body": "Replacement dispatched, tracking to follow.",
            "sent_at": date(2023, 7, 3),
        },
        {
            "message_id": "M0003",
            "ticket_id": "T0002",
            "author_email": "amara.osei+shop@example.com",
            "body": "Confirming the new address.",
            "sent_at": date(2024, 4, 15),
        },
        {
            "message_id": "M0004",
            "ticket_id": "T0003",
            "author_email": "support@shop.example",
            "body": "Unsubscribed. Confirmation sent to Amara.Osei@Example.com.",
            "sent_at": date(2025, 9, 10),
        },
        # A shared thread: another customer replies inside the subject's ticket.
        {
            "message_id": "M0005",
            "ticket_id": "T0002",
            "author_email": "jonas.berg12@example.com",
            "body": "I share this delivery slot - please keep my address unchanged.",
            "sent_at": date(2024, 4, 16),
        },
        # About the subject, but no key join reaches it: someone else's ticket, subject in the body.
        {
            "message_id": "M0006",
            "ticket_id": "T0009",
            "author_email": "priya.rao8@example.com",
            "body": "My colleague amara.osei@example.com placed this order on my behalf.",
            "sent_at": date(2024, 6, 1),
        },
        {
            "message_id": "M0007",
            "ticket_id": "T0014",
            "author_email": "support@shop.example",
            "body": "Escalated. Refund was actually issued to amara.osei@example.com in error.",
            "sent_at": date(2025, 2, 18),
        },
    ]
    for i, ticket in enumerate(support_tickets[3:], start=8):
        ticket_messages.append(
            {
                "message_id": f"M{i:04d}",
                "ticket_id": str(ticket["ticket_id"]),
                "author_email": str(ticket["requester_email"]),
                "body": "Thanks for the update.",
                "sent_at": ticket["created_at"],
            }
        )

    marketing_contacts: List[Row] = [
        {
            "contact_id": "MC001",
            "email_lower": "amara.osei@example.com",
            "consent": "withdrawn",
            "source": "checkout opt-in 2023-03-14",
        }
    ]
    for i, customer in enumerate(customers[1:40], start=2):
        marketing_contacts.append(
            {
                "contact_id": f"MC{i:03d}",
                "email_lower": str(customer["email"]).lower(),
                "consent": rng.choice(["granted", "granted", "withdrawn"]),
                "source": "checkout opt-in",
            }
        )

    identity_links: List[Row] = [
        {
            "link_id": "L001",
            "anon_id": "anon-9f3c21",
            "customer_id": "C0001",
            "linked_at": date(2023, 3, 14),
            "method": "login on same device",
        },
        {
            "link_id": "L002",
            "anon_id": "anon-4b70de",
            "customer_id": "C0001",
            "linked_at": date(2024, 4, 11),
            "method": "email click-through",
        },
    ]
    for i, customer in enumerate(customers[1:25], start=3):
        identity_links.append(
            {
                "link_id": f"L{i:03d}",
                "anon_id": f"anon-{rng.randrange(16 ** 6):06x}",
                "customer_id": str(customer["customer_id"]),
                "linked_at": date(2024, 1, 1) + timedelta(days=rng.randint(0, 500)),
                "method": "login on same device",
            }
        )

    web_events: List[Row] = []
    ev = 0
    for _ in range(22):  # logged in
        ev += 1
        web_events.append(
            {
                "event_id": f"E{ev:05d}",
                "anon_id": None,
                "customer_id": "C0001",
                "path": rng.choice(["/cart", "/product/P012", "/account", "/checkout"]),
                "ts": date(2024, 1, 1) + timedelta(days=rng.randint(0, 800)),
            }
        )
    for anon in ("anon-9f3c21", "anon-4b70de"):  # pre-login, same person
        for _ in range(7):
            ev += 1
            web_events.append(
                {
                    "event_id": f"E{ev:05d}",
                    "anon_id": anon,
                    "customer_id": None,
                    "path": rng.choice(["/", "/search", "/product/P031"]),
                    "ts": date(2023, 3, 1) + timedelta(days=rng.randint(0, 400)),
                }
            )
    for customer in customers[1:30]:
        for _ in range(rng.randint(1, 6)):
            ev += 1
            web_events.append(
                {
                    "event_id": f"E{ev:05d}",
                    "anon_id": None,
                    "customer_id": str(customer["customer_id"]),
                    "path": "/",
                    "ts": date(2024, 6, 1) + timedelta(days=rng.randint(0, 400)),
                }
            )

    referrals: List[Row] = [
        {
            "referral_id": "R001",
            "referrer_customer_id": "C0001",
            "referee_email": "yuki.sato21@example.com",
            "created_at": date(2024, 2, 3),
            "reward": "SGD 10",
        },
        {
            "referral_id": "R002",
            "referrer_customer_id": "C0019",
            "referee_email": "amara.osei@example.com",
            "created_at": date(2023, 3, 10),
            "reward": "SGD 10",
        },
    ]
    for i, customer in enumerate(customers[3:18], start=3):
        referrals.append(
            {
                "referral_id": f"R{i:03d}",
                "referrer_customer_id": str(customer["customer_id"]),
                "referee_email": str(rng.choice(customers)["email"]).lower(),
                "created_at": date(2024, 1, 1) + timedelta(days=rng.randint(0, 500)),
                "reward": "SGD 10",
            }
        )

    return {
        "customers": customers,
        "orders": orders,
        "order_items": order_items,
        "payments": payments,
        "products": products,
        "support_tickets": support_tickets,
        "ticket_messages": ticket_messages,
        "marketing_contacts": marketing_contacts,
        "web_events": web_events,
        "identity_links": identity_links,
        "referrals": referrals,
    }


# --------------------------------------------------------------------------------------
# 8. Reporting
# --------------------------------------------------------------------------------------


def coverage(corpus: Corpus, hits: Sequence[Hit], seed_email: str = SUBJECT_EMAIL) -> Dict[str, int]:
    naive = naive_extract(corpus, seed_email)
    full = {(h.table, h.pk) for h in hits}
    return {
        "naive": len(naive),
        "resolved": len(full),
        "missed_by_naive": len(full - naive),
        "mention_only": len([h for h in hits if h.how == MENTION]),
        "shared": len([h for h in hits if h.shared]),
    }


def summarize(corpus: Corpus, ident: Identity, hits: Sequence[Hit]) -> str:
    lines: List[str] = []
    cov = coverage(corpus, hits)
    lines.append("IDENTITY")
    for key in ident.keys:
        flag = " " if key.usable else "!"
        lines.append(f" {flag} {key.strength:<7} {key.key_type:<12} {key.value:<34} {key.evidence}")
    if ident.variants:
        lines.append("")
        lines.append("STORED SPELLINGS OF THE SAME MAILBOX (what an exact match misses)")
        for note in ident.variants:
            lines.append(f"   {note}")
    lines.append("")
    lines.append("COVERAGE")
    lines.append(f"   naive query found         {cov['naive']:>4} rows")
    lines.append(f"   resolved extract found    {cov['resolved']:>4} rows")
    lines.append(f"   missed by naive           {cov['missed_by_naive']:>4} rows")
    lines.append(f"   reachable only by text    {cov['mention_only']:>4} rows")
    lines.append(f"   shared with a third party {cov['shared']:>4} rows (redacted before disclosure)")
    lines.append("")
    lines.append("PER TABLE")
    by_table: Dict[str, List[Hit]] = {}
    for hit in hits:
        by_table.setdefault(hit.table, []).append(hit)
    for table in sorted(by_table):
        group = by_table[table]
        how: Dict[str, int] = {}
        for hit in group:
            how[hit.how] = how.get(hit.how, 0) + 1
        detail = ", ".join(f"{k}={v}" for k, v in sorted(how.items()))
        shared = sum(1 for h in group if h.shared)
        suffix = f", {shared} redacted" if shared else ""
        lines.append(f"   {table:<20} {len(group):>4}  ({detail}{suffix})")
    return "\n".join(lines)


if __name__ == "__main__":
    corpus = build_corpus()
    ident = resolve_identity(corpus, SUBJECT_EMAIL)
    hits = extract(corpus, ident)
    print(summarize(corpus, ident, hits))
    print()
    print("OVER-COLLECTION (naive FK sweep):", naive_fk_sweep(corpus, hits))
    print("WEAK-LINK COST (if auto-included):", weak_link_cost(corpus, ident))
    print()
    plan = erasure_plan(hits, corpus)
    counts: Dict[str, int] = {}
    for action in plan:
        counts[action.action] = counts.get(action.action, 0) + 1
    print("ERASURE PLAN:", counts)
