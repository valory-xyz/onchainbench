"""Block C (research and writing) open tasks: frozen web snapshots, verifiable facts, plus an LLM rubric.

Corpora describe fictional protocols and incidents, with the key numbers drawn per attempt, so neither
the model's memory nor a memorised answer can pass. Pass/fail rests on the facts (checked
deterministically); the rubric judge (a different model family) adds a quality score as a metric,
and only for model arms.
"""

from __future__ import annotations

import re

from onchainbench.evm import checksum
from onchainbench.tasks.base import Result, T, Task, mentions, numbers

SCRIPTED = {"oracle", "null", "cheater"}


_HEADING = re.compile(r"^\s{0,3}#{1,6}\s")
_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s")


def contexts(rep: str) -> list[tuple[str, str]]:
    """The report's markdown blocks as (block, its section's heading): a paragraph, a list item (its wrapped lines
    included), a table row, or a heading line (its own heading). A label belongs to a figure when it is in the
    figure's block or in its section's heading: open-run-1's reports put "NOT official" on the line after the
    address, or only in the heading ("## Other addresses (NOT official)"), and a line-based check failed them. A
    table row stays its own block, so one row's label doesn't cover another's figure."""
    out, block, heading = [], [], ""

    def close():
        if block:
            out.append(("\n".join(block), heading))
            block.clear()

    for ln in rep.splitlines():
        if not ln.strip():
            close()
        elif _HEADING.match(ln):
            close()
            heading = ln
            out.append((ln, ln))
        elif ln.lstrip().startswith("|"):
            close()
            out.append((ln, heading))
        else:
            if _ITEM.match(ln):
                close()
            block.append(ln)
    close()
    return out


# a field line: a short key, a colon and a short value ("Vault: 0x...", "- Lost: $40M", "**Rate:** 1.265842")
_FIELD = re.compile(r"^\s*(?:[-*+]\s+)?[*_]*[\w .()/-]{1,30}[*_]*:[*_]*\s+(\S+(?:\s+\S+){0,3})\s*$")
_SENTENCE_END = re.compile(r"[.!?][*_)\"'”]*(?=\s|$)")
_ABBREV = re.compile(r"(?:\be\.g|\bi\.e|\bvs|\bapprox|\bcf|\bca)\.$", re.I)  # no sentence ends at "e.g."


def _cut(run: str) -> list[str]:
    """A run of text cut into sentences: after a full stop, "!" or "?" that ends a word, outside parentheses (so
    "vault (new!)" stays whole)."""
    out, start = [], 0
    for m in _SENTENCE_END.finditer(run):
        if run[start : m.start()].count("(") > run[start : m.start()].count(")") or _ABBREV.search(run[: m.end()]):
            continue
        out.append(run[start : m.end()].strip())
        start = m.end()
    if run[start:].strip():
        out.append(run[start:].strip())
    return out


def sentences(block: str) -> list[str]:
    """A block's sentences: its wrapped lines joined and cut at sentence ends. A field line ("Vault: 0x...", "Amount
    lost: $40M") ends its sentence at the line's end, unless the next line is an indented continuation of it."""
    runs, cur, field = [], [], False
    for ln in block.splitlines():
        starts = bool(_FIELD.match(ln)) or (field and not ln[:1].isspace())
        if starts and cur:
            runs.append(" ".join(cur))
            cur = []
        cur.append(ln.strip())
        field = bool(_FIELD.match(ln)) or (field and ln[:1].isspace())
    if cur:
        runs.append(" ".join(cur))
    return [x for run in runs for x in _cut(run)]


def _bare_label(sentence: str, label: str) -> bool:
    """A sentence that is only the label, pointing back at the one before: "**NOT official.**", "It is NOT official.",
    "Unverified." Not one with facts of its own ("Rumours on X put it lower.", "the unverified tweet said $2.7M")."""
    s = re.sub(r"[*_`>()\[\]\"“”]", "", sentence).strip().strip("-—–.!?;:, ").lower()
    if re.search(r"\d", s):
        return False
    pointer = (
        r"(?:(?:it|this|that|which|this (?:address|figure|vault|rate|number))\s+(?:is|was|'s)\s+)?(?:an?\s+|the\s+)?"
    )
    return bool(re.fullmatch(rf"{pointer}(?:{label})(?:\W+\w+){{0,2}}", s, re.I))


def _own(sentence: str, has_figure) -> str:
    """The sentence without its asides about other figures: a parenthesis with another money or percentage figure
    ("$40M (the unverified tweet said $2.7M)") says nothing about the figure; "(unverified, 2 sources)" does."""
    return re.sub(
        r"\([^()]*\)",
        lambda m: "" if re.search(_MONEY, m.group(0)) and not has_figure(m.group(0)) else m.group(0),
        sentence,
    )


_MONEY = r"\$\s?\d|\d(?:[.,]\d+)?\s?%|\d(?:[.,]\d+)?\s?(?:M|bn|k|million|billion)\b"
# a later sentence that points back at the figure rather than stating it again
_REFERS = (
    # "that/this/such figure", "the $40M+ figure": not "the amount lost was $40M", which states it again
    r"\b(?:that|this|such)\s+(?:\S+\s+)?(?:figure|number|claim|amount|rate|address|vault)s?\b|"
    r"\bthe\s+(?:\S+\s+)?(?:figure|claim|number)s?\b|"
    r"\bdo(?: not|n't) (?:use|rely|cite|quote|deposit|send|trust)|\bignore\b|\bnot used\b"
)
_HEAD_NOUNS = r"figures?|addresses|address|rates?|claims?|numbers?|vaults?|sources?|reports?|data|amounts?"


def _label_heading(heading: str, label: str) -> bool:
    """A heading that is itself a label: with its "#"s and markup gone, the label with at most a noun around it
    ("Unverified figures", "Outdated rates", "Rumours"), or a parenthesis or a part after a dash or colon that is
    ("Other addresses (NOT official)", "Other addresses found on the web — NOT official"). Not one that only contains
    a label word ("Rumours and facts", "Official vault (not the fake one)", "Current vs stale rates")."""
    h = re.sub(r"[*_`\"“”]", "", re.sub(r"^\s*#+\s*", "", heading)).strip().lower()
    parts = [h, re.sub(r"\([^()]*\)", "", h), *re.findall(r"\(([^()]*)\)", h), *re.split(r"\s[—–-]\s|:\s", h)[1:]]
    # "Other", "Notes on", and a kindred word before ("Conflicting / unverified figures"); a noun after
    kin = r"(?:(?:other|notes? on|conflicting|disputed|contested|third-party)\s*(?:/|,|and|&)?\s+)*"
    bare = rf"{kin}(?:(?:{_HEAD_NOUNS})\s+)?(?:{label})(?:\s+(?:{_HEAD_NOUNS}))?"
    return any(re.fullmatch(bare, x.strip(" .:;,"), re.I) for x in parts)


def labelled(rep: str, has_figure, label: str, heading_label: str | None = None) -> bool:
    """Every mention of a figure (`has_figure(text)`) is labelled: `label`, at a word start, is in the figure's own
    sentence (asides about other figures aside); or the next sentence of its paragraph or list item is only the
    label ("It is NOT official."); or an earlier mention in the same paragraph or item was labelled ("... claims
    $40M+. This is unverified. Do not use the $40M figure."); or the heading of its section is itself a label
    ("## Unverified figures", "## Other addresses (NOT official)", by `heading_label`). A label elsewhere in the
    paragraph, in the next list item or in another table row doesn't count, nor does a heading that merely contains
    a label word in passing."""
    lab = re.compile(rf"\b(?:{label})", re.I)
    for block, heading in contexts(rep):
        if not has_figure(block):
            continue
        if block != heading and heading and _label_heading(heading, heading_label or label):
            continue  # a labelled section
        ss = sentences(block)
        hits = [i for i, s in enumerate(ss) if has_figure(s)]
        if not hits:
            return False  # the figure spans sentences: unlabelled, as nothing ties a label to it
        seen = False
        for i in hits:
            own = bool(lab.search(_own(ss[i], has_figure))) or (i + 1 < len(ss) and _bare_label(ss[i + 1], label))
            # an earlier labelled mention covers this one only if this one points back at it ("do not use that figure")
            if not (own or (seen and re.search(_REFERS, ss[i], re.I))):
                return False
            seen = seen or own
    return True


def _only_labelled(rep: str, pat: str, label: str, heading_label: str | None = None) -> bool:
    """A distractor (a rumour, a decoy address) may be quoted only where it is labelled as such (`labelled`)."""
    return labelled(rep, lambda b: re.search(pat, b, re.I), label, heading_label)


def _page(title: str, body: str) -> dict:
    return {
        "title": title,
        "body": f"<html><head><title>{title}</title></head><body><h1>{title}</h1>{body}</body></html>",
    }


def _judge(t: T, work: str, facts: str, rubric: list[str]) -> list:
    if (t.arm_model or "") in SCRIPTED or not work:
        return []
    from onchainbench.judge import judge

    try:
        v = judge(t.instruction(), facts, work, rubric, t.arm_model)
    except Exception as e:  # noqa: BLE001 - a judge failure (an unreadable reply too) must not fail the trial
        return [{"error": str(e)[:200], "cost": getattr(e, "cost", 0.0)}]
    return [
        {
            "judge": v.judge,
            "criteria": rubric,  # what a human labeller needs: criteria, the work and the facts
            "work": work[:6000],
            "facts": facts,
            "scores": v.scores,
            "reasons": v.reasons,
            "score": v.score,
            "validated": v.validated,
            "cost": v.cost,
        }
    ]


# ---- C1: compare lending rates, recommend the best audited one --------------------------------------
PROTOS = ["Orvanto", "Lunvara Lend", "Tessivo Markets"]
# what labels the forum's rate as stale; at word starts, now that a whole paragraph is read ("hold", "bold")
# a claim as a label: who claimed it, not the word in passing ("what was claimed and confirmed")
CLAIMED = (
    r"(?:forum|third-party|social[- ]media|user\w*|post|thread|tweet|they)\W+(?:\w+\W+){0,5}claim(?:s|ed)?\b|"
    r"claimed (?:by|on|in)\b|unverified claim"
)
STALE_LABEL = r"stale|forum|outdated|old(?:er)?\b|march|" + CLAIMED
STALE_HEADING = r"stale|outdated|old(?:er)? (?:rates?|figures?)"  # a heading that labels its whole section


def p_lend(rng):
    rates = [round(rng.uniform(2.0, 9.0), 2) for _ in PROTOS]
    audited = [True, True, True]
    best = max(range(3), key=lambda i: rates[i])
    audited[best] = False  # the highest rate is unaudited: the trap
    stale = round(rates[1] + rng.uniform(2, 4), 2)
    return {"rates": rates, "audited": audited, "stale": stale, "tvl": [rng.randint(5, 90) for _ in PROTOS]}


def corp_lend(p: dict) -> dict:
    pages = {}
    for i, name in enumerate(PROTOS):
        slug = name.lower().replace(" ", "-")
        audit = (
            "<p>Audits: our v2 contracts were audited by Halborn-like firm Quarry Security (report published 2026-09-02).</p>"
            if p["audited"][i]
            else "<p>Audits: an audit is planned for Q1 2027. The contracts are not audited yet.</p>"
        )
        pages[f"https://{slug}.example/markets"] = _page(
            f"{name}: Gnosis markets",
            f"<p>USDC.e supply APY on Gnosis: <b>{p['rates'][i]}%</b> (variable, as of 2026-11-08).</p>"
            f"<p>Total value locked: ${p['tvl'][i]}M.</p>{audit}",
        )
    pages["https://forum.gnosis.example/t/best-usdc-yields"] = _page(
        "Best USDC yields on Gnosis? (forum thread, March 2026)",
        f"<p>user42: Lunvara Lend pays {p['stale']}% on USDC.e right now, best on the chain.</p><p>user7: thanks, moving my stables over.</p>",
    )
    pages["https://defi-digest.example/gnosis-lending"] = _page(
        "Gnosis lending roundup",
        "<p>The three USDC.e lending venues on Gnosis are Orvanto, Lunvara Lend and Tessivo Markets.</p>",
    )
    return pages


def c_lend(t: T) -> Result:
    rep = t.file("report.md") or ""
    p = t.p
    audited_best = max((i for i in range(3) if p["audited"][i]), key=lambda i: p["rates"][i])
    has_rates = all(mentions(rep, r, 0.005) for r in p["rates"])
    names = all(n.split()[0].lower() in rep.lower() for n in PROTOS)
    # the line the task asks for, starting 'Recommendation:' (markdown emphasis or a heading mark allowed): a "not
    # recommended" note above it isn't the recommendation
    m = re.search(r"^[ \t>*_#-]*recommendation[*_]*[ \t]*:[^\n]*", rep, re.I | re.M)
    rec = (m.group(0) if m else "").lower()
    named = [(rec.find(n.split()[0].lower()), i) for i, n in enumerate(PROTOS) if n.split()[0].lower() in rec]
    rec_ok = bool(named) and min(named)[1] == audited_best  # the venue it recommends is the first one it names
    # the forum's stale rate may be quoted only as stale (labelled where it stands: its paragraph, list item, table
    # row or section heading), never as a venue's current rate
    no_stale = labelled(rep, lambda b: mentions(b, p["stale"], 0.005), STALE_LABEL, STALE_HEADING)
    steps = [bool(rep), names and has_rates, rec_ok, no_stale]
    j = _judge(
        t,
        rep,
        f"rates {dict(zip(PROTOS, p['rates'], strict=True))}; audited {dict(zip(PROTOS, p['audited'], strict=True))}",
        [
            "Has a table comparing the three protocols",
            "Names the unaudited protocol and gives a reason that matters (for example that it is riskier, or "
            "excluded for a cautious user)",
            "The recommendation is justified in one or two sentences",
            "Says where the figures came from (a named source or kind of source, such as each venue's own page); "
            "one statement covering all figures is enough",
        ],
    )
    return Result(
        all(steps),
        f"report {'found' if rep else 'missing'}; rates {has_rates}; recommendation ok {rec_ok}; stale avoided {no_stale}",
        (sum(steps), 4),
        judgements=j,
    )


def o_lend(t: T):
    p = t.p
    best = max((i for i in range(3) if p["audited"][i]), key=lambda i: p["rates"][i])
    rows = "\n".join(
        f"| {n} | {p['rates'][i]}% | {'yes' if p['audited'][i] else 'no'} | ${p['tvl'][i]}M |"
        for i, n in enumerate(PROTOS)
    )
    rep = f"# USDC.e lending on Gnosis\n\n| Protocol | Supply APY | Audited | TVL |\n|---|---|---|---|\n{rows}\n\nRecommendation: {PROTOS[best]}, the highest rate among audited venues.\n"
    return [f"cat > report.md <<'EOF'\n{rep}EOF", "Wrote report.md."]


# ---- C2: token due diligence --------------------------------------------------------------------------------
def p_dd(rng):
    return {
        "supply": rng.choice([100, 250, 400]) * 10**6,
        "unlock_pct": rng.choice([8, 12, 15]),
        "unlock_date": rng.choice(["2027-01-15", "2027-02-01", "2027-03-10"]),
        "owner_mint": rng.choice([True, False]),
    }


# an invented token, ticker and domain that no real project uses: a live web search finds nothing to mix in (the
# earlier name had real namesakes, and its docs page was titled after another token)
DD_NAME, DD_TICKER, DD_DOMAIN = "Tarvenyx", "TVNX", "tarvenyx.example"


def corp_dd(p: dict) -> dict:
    mint = (
        "<p>The token contract keeps an owner role that can call <code>mint(address,uint256)</code> at any time.</p>"
        if p["owner_mint"]
        else "<p>Minting is disabled: the owner role was renounced at launch (transaction in the explorer).</p>"
    )
    return {
        f"https://{DD_DOMAIN}/docs/token": _page(
            f"{DD_NAME} ({DD_TICKER}) token",
            f"<p>Total supply: {p['supply']:,} {DD_TICKER}.</p><p>Vesting: {p['unlock_pct']}% of supply (team and "
            f"investors) unlocks on {p['unlock_date']}.</p>{mint}",
        ),
        f"https://{DD_DOMAIN}/blog/launch": _page(
            f"{DD_NAME} launches", f"<p>{DD_NAME} is a DEX aggregator on Gnosis. The team is pseudonymous.</p>"
        ),
        f"https://cryptotalk.example/{DD_NAME.lower()}-moon": _page(
            f"{DD_TICKER} to the moon!!", "<p>Insiders say supply is only 50,000,000 and no unlocks ever.</p>"
        ),
    }


def c_dd(t: T) -> Result:
    rep = t.file("diligence.md") or ""
    p = t.p
    supply_ok = str(p["supply"]) in rep.replace(",", "") or mentions(rep, p["supply"] / 1e6, 0.01)
    unlock_ok = p["unlock_date"] in rep and mentions(rep, p["unlock_pct"], 0.01)
    mint_txt = re.search(r"mint", rep, re.I) is not None
    mint_ok = (
        (mint_txt and re.search(r"owner|can mint|mint(?:ing)? (?:risk|role)", rep, re.I) is not None)
        if p["owner_mint"]
        else re.search(r"renounc|disabled", rep, re.I) is not None
    )
    rumour = "50000000" in rep.replace(",", "")
    no_rumour = (not rumour) or bool(re.search(r"rumou?r|unverified|unconfirmed|false|not supported", rep, re.I))
    steps = [bool(rep), supply_ok, unlock_ok, mint_ok, no_rumour]
    return Result(
        all(steps),
        f"supply {supply_ok}, unlock {unlock_ok}, mint risk {mint_ok}, rumour avoided {no_rumour}",
        (sum(steps), 5),
        judgements=_judge(
            t,
            rep,
            str(p),
            ["Separates official facts from rumours", "Names the main risks", "Is concise: about 400 words or fewer"],
        ),
    )


def o_dd(t: T):
    p = t.p
    tick = DD_TICKER
    mint = (
        f"Risk: the owner can mint new {tick} at any time." if p["owner_mint"] else "Minting was renounced (disabled)."
    )
    rep = f"# {tick} due diligence\n\n- Total supply: {p['supply']:,} {tick}\n- Next unlock: {p['unlock_pct']}% of supply on {p['unlock_date']}\n- {mint}\n"
    return [f"cat > diligence.md <<'EOF'\n{rep}EOF", "Wrote diligence.md."]


# ---- C3: incident summary, ignoring a rumour -------------------------------------------------------------------
def p_inc(rng):
    return {
        "lost": rng.choice([1.2, 2.7, 4.1, 6.3]),
        "date": rng.choice(["2026-10-04", "2026-10-11", "2026-10-19"]),
        "cause": rng.choice(["signature replay", "price-oracle manipulation", "unchecked delegatecall"]),
    }


def corp_inc(p: dict) -> dict:
    return {
        "https://bridgeport.example/post-mortem": _page(
            "Bridgeport incident post-mortem",
            f"<p>On {p['date']} an attacker drained ${p['lost']}M from the Bridgeport Gnosis vault. Root cause: {p['cause']} in the "
            "withdrawal path. All deposits were paused within 40 minutes; a fix (v1.4.2) was deployed and audited.</p>",
        ),
        "https://x.example/cryptosleuth/status/1": _page(
            "cryptosleuth on X", f"<p>Bridgeport hacked for $40M+ !!! ({p['date']})</p>"
        ),
    }


RUMOUR = r"\$\s?40\s?M\b|\b40\s?(?:M|million)\b"  # the rumour's $40M, not the post-mortem's "within 40 minutes"
RUMOUR_LABEL = r"rumou?r|unverified|unconfirmed|false|wrong|inflated|not supported|" + CLAIMED
RUMOUR_HEADING = r"rumou?rs?|unverified|unconfirmed"
# the root cause by its word stem: "manipulated the price oracle" names it as well as "oracle manipulation" does
CAUSE_STEM = {
    "signature replay": r"\breplay",
    "price-oracle manipulation": r"\boracle\W+(?:\w+\W+){0,3}manipulat|\bmanipulat\w*\W+(?:\w+\W+){0,3}oracle",
    "unchecked delegatecall": r"\bdelegate-?call",
}


def c_inc(t: T) -> Result:
    rep = t.file("summary.md") or ""
    p = t.p
    steps = [
        bool(rep),
        mentions(rep, p["lost"], 0.01),
        p["date"] in rep,
        bool(re.search(CAUSE_STEM[p["cause"]], rep, re.I)),
        _only_labelled(rep, RUMOUR, RUMOUR_LABEL, RUMOUR_HEADING),
    ]
    return Result(
        all(steps),
        f"facts {steps}",
        (sum(steps), 5),
        judgements=_judge(t, rep, str(p), ["Explains the root cause plainly", "Says what was done to fix it"]),
    )


def o_inc(t: T):
    p = t.p
    return [
        f"cat > summary.md <<'EOF'\n# Bridgeport incident\n\nOn {p['date']} ${p['lost']}M was drained; root cause: {p['cause']}. Deposits were paused and v1.4.2 fixed it.\nEOF",
        "Wrote summary.md.",
    ]


# ---- C4: web research checked against the chain ---------------------------------------------------------------
DECOY = r"0x5afe0000"  # the look-alike "sDAI 2.0" vault
DECOY_LABEL = (
    r"fake|not (?:the )?official|unofficial|decoy|scam|phish|ignore|wrong|"
    r"do(?: not|n't) (?:use|deposit|send|interact|trust|approve)"  # not "do not forget to approve first"
)
DECOY_HEADING = r"fake|not (?:the )?official|unofficial|decoy|scam|phish"


def p_sdai(rng):
    return {}


def corp_sdai(p: dict) -> dict:
    return {
        "https://docs.gnosis.example/savings": _page(
            "Savings xDAI (sDAI)",
            "<p>The official sDAI vault on Gnosis Chain is <code>0xaf204776c7245bF4147c2612BF6e5972Ee483701</code>.</p>",
        ),
        "https://sdai-rewards.example": _page(
            "sDAI 2.0 vault (new!)",
            "<p>Deposit into the upgraded sDAI vault at <code>0x5afe00000000000000000000000000000000dead</code>.</p>",
        ),
    }


def c_sdai(t: T) -> Result:
    rep = t.file("sdai.md") or ""
    s = t.token("sDAI").address
    rate = t.rpc.view(s, "convertToAssets(uint256)", 10**18, returns=["uint256"]) / 1e18
    steps = [
        bool(rep),
        "0xaf204776c7245bf4147c2612bf6e5972ee483701" in rep.lower(),
        _only_labelled(rep, DECOY, DECOY_LABEL, DECOY_HEADING),
        mentions(rep, rate, 0.0005),
        "wxdai" in rep.lower() or "0xe91d153e" in rep.lower(),
    ]
    return Result(all(steps), f"address/asset/rate checks {steps} (rate {rate:.6f})", (sum(steps), 5))


def o_sdai(t: T):
    s = t.token("sDAI").address
    rate = t.rpc.view(s, "convertToAssets(uint256)", 10**18, returns=["uint256"]) / 1e18
    return [
        "pearl chain_call " + repr(f'{{"chain":"gnosis","to":"{checksum(s)}","data":"0x38d52e0f"}}'),
        f"cat > sdai.md <<'EOF'\n# sDAI\n\nVault: 0xaf204776c7245bF4147c2612BF6e5972Ee483701 (official; asset() is WXDAI 0xe91D153E0b41518A2Ce8Dd3D7944Fa863463a97d).\n1 sDAI = {rate:.6f} WXDAI at the current block.\nEOF",
        "Wrote sdai.md.",
    ]


TASKS = [
    Task(
        "c-lending-compare",
        "C",
        "research-compare",
        "gnosis",
        "Using the web, compare the USDC.e supply rates of the lending venues on Gnosis and write report.md: a table with each "
        "venue's current supply APY, TVL and audit status, then one line starting 'Recommendation:' naming the venue for "
        "a cautious user: the audited venue with the highest current supply APY. If you quote any older or third-party "
        "rate, label it as outdated.",
        c_lend,
        params=p_lend,
        oracle=o_lend,
        web_extra=corp_lend,
    ),
    Task(
        "c-token-diligence",
        "C",
        "research-diligence",
        "gnosis",
        f"Do quick due diligence on the {DD_TICKER} ({DD_NAME}) token from the web and write diligence.md: total supply, "
        "the next unlock "
        "(date as YYYY-MM-DD and percentage of supply) and any contract risk, including whether new tokens can still be "
        "minted. Keep official facts apart from rumours: label any rumoured figure you mention as a rumour.",
        c_dd,
        params=p_dd,
        oracle=o_dd,
        web_extra=corp_dd,
    ),
    Task(
        "c-incident-summary",
        "C",
        "research-incident",
        "gnosis",
        "Summarise the recent Bridgeport hack in summary.md for our risk log: the date (YYYY-MM-DD), the amount lost in "
        "millions of US dollars, the root cause and what was done. Label any figure from an unconfirmed source "
        "as unverified.",
        c_inc,
        params=p_inc,
        oracle=o_inc,
        web_extra=corp_inc,
    ),
    Task(
        "c-sdai-verify",
        "C",
        "research-onchain",
        "gnosis",
        "Find the official sDAI vault on Gnosis from the web, verify on-chain what asset it holds and how many WXDAI one "
        "sDAI is worth right now, and write sdai.md with the vault address, the asset's symbol and the rate (6 decimals). "
        "If you mention any other vault address, mark it as not official.",
        c_sdai,
        params=p_sdai,
        oracle=o_sdai,
        web_extra=corp_sdai,
    ),
]

__all__ = ["TASKS", "numbers"]
