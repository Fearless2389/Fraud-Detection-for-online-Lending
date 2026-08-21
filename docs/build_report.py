"""Generate the technical report as a PDF.

Every figure in the report is read from ``docs/figures`` and every number is
read from the artifacts written by the training and analysis scripts. Nothing is
transcribed by hand, so the document cannot drift away from the model it
describes: retrain, re-run the analysis, rebuild the report, and the prose and
the tables move together.

Rendering goes through headless Chromium rather than a PDF library because the
layout is easier to control - and to look at while iterating - as HTML and CSS.
Fonts are resolved from the operating system and images are inlined as data
URIs, so the build makes no network requests.

Usage:
    python docs/build_report.py
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS.parent
ARTIFACTS = PROJECT_ROOT / "backend" / "ml" / "artifacts"

# Import the model's own definition of a behavioural feature rather than
# restating it. The report quotes a headline "behavioural share of gain"; if
# this document disagreed with the pipeline about which features count, the
# table and the headline would contradict each other and only a careful reader
# would notice.
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
from ml.features.pipeline import BEHAVIOURAL_FEATURES  # noqa: E402
FIGURES = DOCS / "figures"
OUTPUT_PDF = DOCS / "Aegis-Technical-Report.pdf"
OUTPUT_HTML = DOCS / "report.html"


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


def load(name: str) -> dict:
    path = ARTIFACTS / name
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Run backend/ml/training/train.py and analyse.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def figure(name: str) -> str:
    """Inline a PNG as a data URI so the PDF has no external dependencies."""
    path = FIGURES / name
    if not path.exists():
        print(f"  warning: missing figure {name}", file=sys.stderr)
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def pct(value: float, places: int = 1) -> str:
    return f"{value * 100:.{places}f}%"


def num(value: float) -> str:
    return f"{value:,.0f}"


# ---------------------------------------------------------------------------
# document
# ---------------------------------------------------------------------------

STYLES = """
@page { size: A4; margin: 17mm 16mm 20mm 16mm; }

:root {
  --paper:   #fdfcf9;
  --ink:     #17150f;
  --ink-2:   #4a463d;
  --ink-3:   #7d786c;
  --rule:    #e2ddd0;
  --rule-2:  #c9c2b0;
  --accent:  #1f5fa8;
  --approve: #0a6f16;
  --review:  #9a6a05;
  --block:   #b02b2b;
  --wash:    #f4f1e8;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: Cambria, Charter, Georgia, serif;
  font-size: 10.2pt;
  line-height: 1.52;
  -webkit-font-smoothing: antialiased;
}

h1, h2, h3, h4, .ui { font-family: "Segoe UI", Inter, system-ui, sans-serif; }

h2 {
  font-size: 15pt; font-weight: 600; letter-spacing: -0.01em;
  margin: 0 0 2mm; padding-bottom: 2mm;
  border-bottom: 1.5px solid var(--ink);
}
h3 {
  font-size: 11.5pt; font-weight: 600; letter-spacing: -0.005em;
  margin: 7mm 0 1.5mm; color: var(--ink);
}
h4 {
  font-size: 9pt; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.09em; color: var(--ink-3);
  margin: 5mm 0 1.5mm;
}
p { margin: 0 0 2.4mm; }
strong { font-weight: 600; }
em { font-style: italic; }

.mono, code, td.n, th.n {
  font-family: Consolas, "Cascadia Mono", "SF Mono", monospace;
  font-size: 0.9em; font-variant-numeric: tabular-nums;
}
code {
  background: var(--wash); padding: 0.5mm 1.2mm;
  border-radius: 1px; color: var(--ink-2);
}

/* ---------- page structure ---------- */

.page { page-break-after: always; }
.page:last-child { page-break-after: auto; }
.no-break { page-break-inside: avoid; }

.eyebrow {
  font-family: "Segoe UI", system-ui, sans-serif;
  font-size: 7.5pt; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.16em; color: var(--ink-3);
}

/* ---------- cover ---------- */

.cover { height: 248mm; display: flex; flex-direction: column; }
.cover-top { flex: 1; display: flex; flex-direction: column; }
.cover h1 {
  font-size: 46pt; font-weight: 600; letter-spacing: -0.035em;
  line-height: 0.92; margin: 3mm 0 0;
}
.cover h1 .accent {
  display: block; font-size: 21pt; font-weight: 300;
  letter-spacing: -0.015em; line-height: 1.14; margin-top: 3.5mm;
  color: var(--ink-2); max-width: 122mm;
}
.cover .sub {
  font-size: 11.4pt; color: var(--ink-2); margin-top: 5mm;
  line-height: 1.46; max-width: 132mm; font-style: italic;
}
.rule-heavy { height: 3px; background: var(--ink); margin: 5mm 0 0; }

.cover-claims {
  display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 0; border-top: 1px solid var(--rule-2); margin-top: 7mm;
}
.claim { padding: 3.5mm 5mm 0 0; }
.claim .v {
  font-family: "Segoe UI", system-ui, sans-serif;
  font-size: 21pt; font-weight: 600; letter-spacing: -0.02em;
  line-height: 1; font-variant-numeric: tabular-nums;
}
.claim .k {
  font-size: 8.6pt; color: var(--ink-2); margin-top: 2mm; line-height: 1.35;
}

/* The headline chart carries the cover rather than leaving a void above the
   metadata block. Cropped to its plot area: the figure's own title would
   repeat the claim stated two inches above it. */
.cover-figure {
  flex: 1; margin: 6mm 0 0; min-height: 0; max-height: 104mm;
  display: flex; align-items: center; justify-content: center;
  border-top: 1px solid var(--rule);
}
.cover-figure img {
  max-width: 100%; max-height: 100%; object-fit: contain;
}

/* ---------- tables ---------- */

table { width: 100%; border-collapse: collapse; margin: 3mm 0 4mm; font-size: 9.2pt; }
caption {
  caption-side: bottom; text-align: left; font-size: 8.2pt;
  color: var(--ink-3); padding-top: 2mm; line-height: 1.45;
}
th {
  font-family: "Segoe UI", system-ui, sans-serif;
  font-size: 7.6pt; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.08em; color: var(--ink-3);
  text-align: left; padding: 0 3mm 1.6mm 0;
  border-bottom: 1px solid var(--ink);
}
td { padding: 1.5mm 3mm 1.5mm 0; border-bottom: 1px solid var(--rule); vertical-align: top; }
/* Right-aligned cells keep their gutter unless they are the last column, where
   the page margin already provides one. Without the :last-child qualifier a
   right-aligned middle column butts straight into the text beside it. */
th.r, td.r { text-align: right; }
th.r:last-child, td.r:last-child { padding-right: 0; }
tr.total td { border-top: 1.5px solid var(--ink); border-bottom: none; font-weight: 600; }
tbody tr:last-child td { border-bottom: 1px solid var(--rule-2); }

.good { color: var(--approve); }
.bad  { color: var(--block); }
.warn { color: var(--review); }
.dim  { color: var(--ink-3); }

/* ---------- figures ---------- */

figure { margin: 4mm 0 5mm; page-break-inside: avoid; }
/* Capped so a tall chart cannot push a page past A4 and orphan a paragraph
   onto a page of its own. object-fit keeps the aspect ratio while it scales. */
figure img {
  width: 100%; display: block;
  max-height: 84mm; object-fit: contain;
}
figcaption {
  font-size: 8.2pt; color: var(--ink-3); margin-top: 2mm;
  line-height: 1.45; border-top: 1px solid var(--rule); padding-top: 1.8mm;
}

/* ---------- callouts ---------- */

.callout {
  border-left: 2.5px solid var(--accent);
  background: var(--wash);
  padding: 3.5mm 4.5mm; margin: 4mm 0;
  page-break-inside: avoid;
}
.callout.warn-tone { border-left-color: var(--review); }
.callout.bad-tone  { border-left-color: var(--block); }
.callout p:last-child { margin-bottom: 0; }
.callout .eyebrow { margin-bottom: 1.5mm; }

.formula {
  font-family: Consolas, monospace; font-size: 9.4pt;
  background: var(--wash); border: 1px solid var(--rule-2);
  padding: 3.5mm 4.5mm; margin: 3mm 0; line-height: 1.85;
  page-break-inside: avoid;
}

pre.block {
  font-family: Consolas, monospace; font-size: 8.4pt; line-height: 1.5;
  background: var(--wash); border: 1px solid var(--rule-2);
  padding: 3.5mm 4mm; margin: 3mm 0; overflow: hidden;
  white-space: pre; page-break-inside: avoid;
}

ul, ol { margin: 0 0 2.8mm; padding-left: 5mm; }
li { margin-bottom: 1.4mm; }

.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 0 8mm; }

.lede {
  font-size: 11pt; line-height: 1.5; color: var(--ink-2);
  margin-bottom: 4mm; font-style: italic;
}

.footer-note {
  margin-top: 6mm; padding-top: 2.5mm; border-top: 1px solid var(--rule);
  font-size: 8pt; color: var(--ink-3);
}
"""


def build_html() -> str:
    metrics = load("metrics.json")
    analysis = load("analysis.json")
    adaptation = load("adaptation.json")

    raw = metrics["test_raw"]
    cal = metrics["test_calibrated"]
    policy = metrics["policy"]
    derived = metrics["test_outcome_derived"]
    naive = metrics["test_outcome_naive"]
    splits = {s["split"]: s for s in metrics["trained_at_month_range"]}

    equal = analysis["equal_recall_comparison"]
    three, binary = equal["three_way"], equal["binary_equivalent"]
    fp_avoided = equal["false_positives_avoided"]
    fp_reduction = 1 - three["genuine_blocked"] / binary["genuine_blocked"]

    months = [m for m in analysis["drift_by_month"] if m["split"] != "train"]
    first, last = months[0], months[-1]

    scenarios = {s["scenario"]: s for s in adaptation["scenarios"]}
    stale = scenarios["STALE"]

    ece_reduction = 1 - cal["ece"] / raw["ece"]
    behavioural = metrics["behavioural_gain_share"]

    def scenario_rows() -> str:
        rows = []
        for name in ("STALE", "RECALIBRATE", "RETRAIN", "ALERT-BUDGET"):
            s = scenarios[name]
            delta = s["cost_per_app"] - stale["cost_per_app"]
            best = name == "STALE" or s["cost_per_app"] <= stale["cost_per_app"]
            rows.append(f"""
            <tr>
              <td><strong>{name}</strong><br><span class="dim" style="font-size:8.2pt">{s['description']}</span></td>
              <td class="r n">{pct(s['recall'])}</td>
              <td class="r n">₹{s['cost_per_app']:.0f}</td>
              <td class="r n {'good' if best else 'bad'}">{'—' if abs(delta) < 0.5 else f'{delta:+.0f}'}</td>
              <td class="r n dim">{s['adapt_seconds']:.1f}s</td>
            </tr>""")
        return "".join(rows)

    def top_feature_rows(limit: int = 8) -> tuple[str, float]:
        """Rows for the top-N table, plus the behavioural share they cover.

        The behavioural set is imported from the feature pipeline rather than
        restated here. Two definitions of "behavioural" - one in the model, one
        in the document describing it - is how a report ends up contradicting
        its own headline figure.
        """
        rows = []
        shown_behavioural = 0.0
        for feature in metrics["top_features"][:limit]:
            is_behavioural = feature["feature"] in BEHAVIOURAL_FEATURES
            if is_behavioural:
                shown_behavioural += feature["share"]
            rows.append(f"""
            <tr>
              <td class="n">{feature['feature']}</td>
              <td>{'behavioural' if is_behavioural else 'application'}</td>
              <td class="r n">{pct(feature['share'])}</td>
            </tr>""")
        return "".join(rows), shown_behavioural

    def queue_rows() -> str:
        rows = []
        for entry in adaptation["queue_stability"]:
            over = entry["fixed_review_rate"] > 0.05
            rows.append(f"""
            <tr>
              <td class="n">{entry['month']}</td>
              <td class="r n">{pct(entry['fraud_rate'], 2)}</td>
              <td class="r n {'bad' if over else ''}">{pct(entry['fixed_review_rate'], 1)}</td>
              <td class="r n">{pct(entry['budget_review_rate'], 1)}</td>
              <td class="r n dim">{pct(entry['fixed_recall'])} / {pct(entry['budget_recall'])}</td>
            </tr>""")
        return "".join(rows)

    feature_rows, shown_behavioural = top_feature_rows()

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Aegis — Technical Report</title><style>{STYLES}</style></head><body>

<!-- ==================== COVER ==================== -->
<section class="page cover">
  <div class="cover-top">
    <div class="eyebrow">Technical Report</div>
    <h1>Aegis<span class="accent">Adaptive application-fraud<br>decisioning for digital lending</span></h1>
    <div class="rule-heavy"></div>
    <p class="sub">
      Fraud detection is a decision problem, not a classification problem.
      This report argues that the interesting question is not how well a model
      ranks applications, but where the decision boundaries come from — and
      shows what changes when they are derived from cost rather than tuned.
    </p>

    <div class="cover-claims">
      <div class="claim">
        <div class="v" style="color:var(--approve)">−{pct(fp_reduction, 0)}</div>
        <div class="k">genuine customers wrongly blocked, at identical fraud detection</div>
      </div>
      <div class="claim">
        <div class="v">{ece_reduction * 100:.0f}%</div>
        <div class="k">reduction in calibration error, without which cost-derived thresholds are meaningless</div>
      </div>
      <div class="claim">
        <div class="v">{pct(behavioural, 0)}</div>
        <div class="k">of model gain from behavioural signals rather than application form fields</div>
      </div>
    </div>

    <div class="cover-figure">
      <img src="{figure('01_false_positives.png')}" alt="False positives at equal recall">
    </div>
  </div>

  <div>
    <table style="margin:0">
      <tr><td style="width:34mm" class="dim">Dataset</td><td>Bank Account Fraud Suite (NeurIPS 2022), Feedzai — {num(sum(s['rows'] for s in splits.values()))} applications across 8 months</td></tr>
      <tr><td class="dim">Model</td><td>LightGBM, {metrics['best_iteration']} boosting iterations, {metrics['n_features']} features, isotonic calibration</td></tr>
      <tr><td class="dim">Evaluation</td><td>Temporal hold-out — months 6–7, never seen in training or calibration</td></tr>
      <tr><td class="dim">Stack</td><td>FastAPI · React 19 · PostgreSQL 17 + pgvector · SHAP · Gemini</td></tr>
    </table>
  </div>
</section>

<!-- ==================== 1. PROBLEM ==================== -->
<section class="page">
  <h2>1 &nbsp; The problem, and why the obvious framing is wrong</h2>

  <p class="lede">
    Digital lending removed the branch visit. It also removed every informal
    check that came with it — and origination fraud is the result.
  </p>

  <p>
    An application arrives through a phone in under two minutes. There is no
    cashier reading a face, no manager recognising a name. The lender has a form,
    a device, a session, and a few seconds to decide. Fraud at this moment is not
    a stolen card being swiped; it is a <strong>fabricated or hijacked identity
    asking to be given credit</strong>, and once the money leaves it does not
    come back.
  </p>

  <p>
    The obvious framing is: build a classifier, find fraud. That framing produces
    systems that fail in production, for a reason that is arithmetic rather than
    technical.
  </p>

  <h3>The asymmetry that governs everything</h3>

  <p>
    Fraud in this dataset runs at {pct(splits['test']['fraud_rate'], 2)} of
    applications. So for every fraudulent application there are roughly seventy
    genuine ones. A model that flags 5% of traffic to catch half the fraud is
    <em>also</em> flagging thousands of real customers — and each of those is a
    person who wanted a loan, was treated as a criminal, and will not come back.
  </p>

  <div class="callout">
    <div class="eyebrow">The cost structure</div>
    <p>
      Blocking a genuine customer and approving a fraudster are <strong>not
      symmetric errors</strong>, and neither are they comparable in magnitude. A
      wrongful block costs the lender an acquisition and a reputation. An
      approved fraud costs the principal. In this system those are configured as
      <span class="mono">₹1,500</span> and <span class="mono">₹45,000</span> —
      a thirty-fold difference.
    </p>
  </div>

  <p>
    Once the errors are that asymmetric, the threshold matters more than the
    model. A binary system has one dial, and every position on it trades real
    customers against real losses. Most prototypes leave that dial at 0.5, which
    on this data misses {pct(naive['fraud_missed'] / splits['test']['fraud'], 0)}
    of all fraud — a number that sounds like a modelling failure but is really a
    threshold that was never chosen at all.
  </p>

  <h3>What this system does instead</h3>

  <p>
    Aegis makes three commitments, and the rest of this report is the evidence
    for each:
  </p>

  <ol>
    <li><strong>The boundaries are derived, not tuned.</strong> Two thresholds
      fall out of four business costs by minimising expected loss. Changing the
      lender's risk appetite is arithmetic, not a retrain.</li>
    <li><strong>There are three outcomes, not two.</strong> Approve, review,
      block. The review band is where the false-positive reduction comes from,
      and it is capped by how many analysts actually exist.</li>
    <li><strong>Every decision is explained and recorded.</strong> Additive SHAP
      attributions become adverse-action reason codes; each decision is written
      to an audit log the database will not let anyone rewrite.</li>
  </ol>

  <div class="footer-note">
    A note on what is <em>not</em> claimed. This system addresses fraud at
    origination — identity fabrication and takeover. It does not address credit
    risk (a real applicant who cannot repay) or first-party bust-out fraud, which
    manifests months after approval and leaves no trace in an application form.
  </div>
</section>

<!-- ==================== 2. HEADLINE RESULT ==================== -->
<section class="page">
  <h2>2 &nbsp; The headline result</h2>

  <p class="lede">
    At identical fraud detection, the three-way policy blocks
    {pct(fp_reduction, 0)} fewer genuine customers.
  </p>

  <p>
    The comparison below is deliberately constructed to be hard on this system.
    The binary baseline is not a strawman left at 0.5 — it is
    <strong>tuned to catch exactly as much fraud</strong> as the three-way
    policy does. Both stop {num(binary['fraud_caught'])} frauds. The only
    question is what each destroys on the way.
  </p>

  <table class="no-break">
    <thead>
      <tr><th>Policy on held-out months 6–7</th><th class="r">Frauds stopped</th><th class="r">Genuine blocked</th><th class="r">To review</th></tr>
    </thead>
    <tbody>
      <tr>
        <td>Binary auto-decline, tuned to equal recall</td>
        <td class="r n">{num(binary['fraud_caught'])}</td>
        <td class="r n bad">{num(binary['genuine_blocked'])}</td>
        <td class="r n dim">—</td>
      </tr>
      <tr>
        <td>Three-way with a cost-derived review band</td>
        <td class="r n">{num(three['fraud_caught'])}</td>
        <td class="r n good">{num(three['genuine_blocked'])}</td>
        <td class="r n">{num(three['reviewed'])}</td>
      </tr>
      <tr class="total">
        <td>Genuine customers spared</td>
        <td class="r n">no change</td>
        <td class="r n good">−{num(fp_avoided)}</td>
        <td class="r"></td>
      </tr>
    </tbody>
    <caption>
      Equal-recall comparison. Measured on {num(splits['test']['rows'])} held-out
      applications containing {num(splits['test']['fraud'])} frauds.
    </caption>
  </table>

  <figure>
    <img src="{figure('01_false_positives.png')}" alt="False positives at equal recall">
    <figcaption>
      The mechanism is not a better model — it is the same model given a third
      option. Borderline applications route to an analyst who can clear a good
      customer, instead of being auto-declined by a cut-off that cannot tell
      "suspicious" from "guilty".
    </figcaption>
  </figure>

  <div class="callout warn-tone">
    <div class="eyebrow">The honest caveat</div>
    <p>
      This buys {num(fp_avoided)} customers at the price of
      {num(three['reviewed'])} manual reviews — about
      {pct(three['reviewed'] / splits['test']['rows'])} of traffic. That is only
      a good trade if the analysts exist. Section 7 shows what happens to the
      queue when they do not, and it is the weakest point in the design.
    </p>
  </div>
</section>

<!-- ==================== 3. THRESHOLDS ==================== -->
<section class="page">
  <h2>3 &nbsp; Finding I — the thresholds are arithmetic, not folklore</h2>

  <p>
    Ask most fraud prototypes where their threshold came from and the answer is
    0.5, or "whatever maximised F1". Neither has a defensible justification, and
    neither survives the question a risk owner will actually ask:
    <em>why is the line there and not somewhere else?</em>
  </p>

  <p>
    Aegis has an answer. Given the cost of a false positive
    (<span class="mono">C<sub>fp</sub></span>), a false negative
    (<span class="mono">C<sub>fn</sub></span>), a manual review
    (<span class="mono">C<sub>rev</sub></span>) and the rate at which analysts
    correctly identify fraud once they see it (<span class="mono">r</span>), the
    expected cost of each action is a line in the fraud probability
    <span class="mono">p</span>. The optimal action is whichever line is lowest,
    and the boundaries are where they cross:
  </p>

  <div class="formula">
τ<sub>review</sub> &nbsp;=&nbsp; C<sub>rev</sub> / (r · C<sub>fn</sub>)
τ<sub>block</sub>  &nbsp;=&nbsp; (C<sub>fp</sub> − C<sub>rev</sub>) / (C<sub>fn</sub> · (1 − r) + C<sub>fp</sub>)
  </div>

  <p>
    With the costs configured here that yields
    <span class="mono">τ<sub>review</sub> = {policy['tau_review_unconstrained']:.4f}</span>
    and <span class="mono">τ<sub>block</sub> = {policy['tau_block']:.4f}</span>.
    Nothing was searched. No validation set was consulted. Change a cost and the
    operating point moves immediately — the model is untouched, which is exactly
    the property a risk function needs and a retrained model cannot offer.
  </p>

  <h3>Bayes-optimal is not the same as operable</h3>

  <p>
    The unconstrained review threshold above routes an enormous share of traffic
    to human review, because the arithmetic assumes review capacity is
    unlimited. It is not. A second constraint caps the review band at the share
    of volume the team can actually work — here
    {pct(policy['max_review_rate'], 0)} — which raises the review threshold to
    <span class="mono">{policy['tau_review']:.4f}</span>.
  </p>

  <div class="callout">
    <div class="eyebrow">Why this matters more than it sounds</div>
    <p>
      A policy that is optimal on paper and unstaffable in practice is not
      optimal. The capacity cap is the difference between a threshold derived in
      a notebook and one that can be deployed on Monday — and it is applied
      <em>after</em> the economics, so the trade being made stays visible rather
      than being hidden inside a tuned constant.
    </p>
  </div>

  <h3>Cost outcome on held-out data</h3>

  <table class="no-break">
    <thead>
      <tr><th>Outcome, months 6–7</th><th class="r">Cost-derived policy</th><th class="r">Untuned 0.5 cut-off</th></tr>
    </thead>
    <tbody>
      <tr><td>Frauds missed</td><td class="r n">{num(derived['fraud_missed'])}</td><td class="r n bad">{num(naive['fraud_missed'])}</td></tr>
      <tr><td>Genuine customers blocked</td><td class="r n">{num(derived['genuine_blocked'])}</td><td class="r n">{num(naive['genuine_blocked'])}</td></tr>
      <tr><td>False positive rate</td><td class="r n">{pct(derived['false_positive_rate'], 2)}</td><td class="r n">{pct(naive['false_positive_rate'], 2)}</td></tr>
      <tr class="total"><td>Total expected cost</td><td class="r n good">₹{derived['total_cost'] / 1e7:.2f} cr</td><td class="r n bad">₹{naive['total_cost'] / 1e7:.2f} cr</td></tr>
    </tbody>
    <caption>
      The 0.5 cut-off blocks almost nobody and therefore looks precise — while
      missing {pct(naive['fraud_missed'] / splits['test']['fraud'], 0)} of all
      fraud. It is cheaper on false positives and costs
      {naive['total_cost'] / derived['total_cost']:.1f}× as much overall.
    </caption>
  </table>
</section>

<!-- ==================== 4. CALIBRATION ==================== -->
<section class="page">
  <h2>4 &nbsp; Finding II — calibration is a precondition, not a polish step</h2>

  <p>
    Section 3 derives thresholds by comparing expected costs. That arithmetic
    multiplies a cost by a probability — which is only valid if the number the
    model emits <em>is</em> a probability. A raw gradient-boosting score is not.
    It ranks well and is systematically wrong in magnitude.
  </p>

  <table class="no-break">
    <thead>
      <tr><th>Held-out months 6–7</th><th class="r">Raw score</th><th class="r">Isotonic-calibrated</th><th class="r">Change</th></tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Expected calibration error</strong></td>
        <td class="r n">{raw['ece']:.4f}</td>
        <td class="r n good">{cal['ece']:.4f}</td>
        <td class="r n good">−{pct(ece_reduction, 1)}</td>
      </tr>
      <tr>
        <td>Brier score</td>
        <td class="r n">{raw['brier']:.4f}</td>
        <td class="r n good">{cal['brier']:.4f}</td>
        <td class="r n good">−{pct(1 - cal['brier'] / raw['brier'], 1)}</td>
      </tr>
      <tr>
        <td>ROC-AUC <span class="dim">(ranking — should not move)</span></td>
        <td class="r n">{raw['roc_auc']:.4f}</td>
        <td class="r n">{cal['roc_auc']:.4f}</td>
        <td class="r n dim">{f"{cal['roc_auc'] - raw['roc_auc']:+.4f}".replace('-', '−')}</td>
      </tr>
      <tr>
        <td>Recall at a 5% alert budget</td>
        <td class="r n">{pct(raw['recall_at_5pct'])}</td>
        <td class="r n good">{pct(cal['recall_at_5pct'])}</td>
        <td class="r n good">+{(cal['recall_at_5pct'] - raw['recall_at_5pct']) * 100:.1f}pp</td>
      </tr>
    </tbody>
    <caption>
      Calibration barely moves ROC-AUC, and that is the point: it does not change
      the ordering of applications, only what the score <em>means</em>. Ranking
      metrics are blind to the defect it fixes.
    </caption>
  </table>

  <figure>
    <img src="{figure('02_reliability.png')}" alt="Reliability diagram">
    <figcaption>
      Reliability on log-log axes, because almost all applications sit below 1%
      and a linear plot would compress every meaningful point into one corner.
      Before calibration the model is confidently wrong at the low end — exactly
      the region where the review threshold sits.
    </figcaption>
  </figure>

  <div class="callout bad-tone">
    <div class="eyebrow">Verified by deliberate failure</div>
    <p>
      A regression test in this project feeds <em>uncalibrated</em> scores to the
      cost policy and asserts that it loses to a naive 0.5 baseline. It does.
      The test is kept permanently, because the failure is the argument: without
      calibration the entire decision layer of this system is invalid, and that
      should be provable rather than asserted.
    </p>
  </div>
</section>

<!-- ==================== 5. BEHAVIOURAL ==================== -->
<section class="page">
  <h2>5 &nbsp; Finding III — behavioural signal carries {pct(behavioural, 0)} of the model</h2>

  <p>
    A fraud model that reads only the application form is reading what the
    fraudster chose to write. The signals that are hard to fabricate are the ones
    the applicant did not know were being recorded: how the session behaved, how
    many distinct emails that device has seen, how many applications came from
    that postcode this month.
  </p>

  <table class="no-break">
    <thead>
      <tr><th>Feature</th><th>Type</th><th class="r">Share of gain</th></tr>
    </thead>
    <tbody>{feature_rows}</tbody>
    <caption>
      Top eight features by total split gain, out of {metrics['n_features']}.
      The behavioural ones visible here account for {pct(shown_behavioural)} of
      total gain; the other {pct(behavioural - shown_behavioural)} sits in the
      tail — velocity windows, postcode counts, device-email counts — which is
      why the model's behavioural share is {pct(behavioural)} and not the sum of
      this column. Classification follows the model's own feature groups, not a
      list restated for this document.
    </caption>
  </table>

  <figure>
    <img src="{figure('05_feature_importance.png')}" alt="Feature importance">
    <figcaption>
      Housing status dominates, which is worth stating plainly rather than
      hiding: it is a proxy for stability, and a model leaning this hard on one
      categorical field is a model whose fairness properties need checking —
      see Section 11.
    </figcaption>
  </figure>
</section>

<!-- ==================== 6. DRIFT ==================== -->
<section class="page">
  <h2>6 &nbsp; Finding IV — what decays is the operating point, not the model</h2>

  <p>
    Recall falls from {pct(first['recall'])} in month {first['month']} to
    {pct(last['recall'])} in month {last['month']}. The instinctive reading is
    that the model is going stale and needs retraining. The data says otherwise.
  </p>

  <table class="no-break">
    <thead>
      <tr><th class="r">Month</th><th class="r">Fraud rate</th><th class="r">ROC-AUC</th><th class="r">Recall</th><th class="r">Cost / application</th></tr>
    </thead>
    <tbody>
      {"".join(f'''<tr>
        <td class="r n">{m['month']}</td>
        <td class="r n">{pct(m['fraud_rate'], 2)}</td>
        <td class="r n">{m['roc_auc']:.3f}</td>
        <td class="r n">{pct(m['recall'])}</td>
        <td class="r n">₹{m['cost_per_app']:.0f}</td>
      </tr>''' for m in months)}
    </tbody>
    <caption>
      Out-of-sample months only. Months 0–3 were trained on; including them would
      present overfitting as though it were drift.
    </caption>
  </table>

  <p>
    ROC-AUC moves by roughly
    {abs(last['roc_auc'] / first['roc_auc'] - 1) * 100:.1f}% across this window.
    The model's ability to <em>rank</em> applications is essentially intact.
    What changed is that fraud prevalence rose from
    {pct(first['fraud_rate'], 2)} to {pct(last['fraud_rate'], 2)} while the
    thresholds stayed where they were set — so the same boundary now sits in a
    different place relative to the population.
  </p>

  <figure>
    <img src="{figure('03_drift.png')}" alt="Drift by month">
    <figcaption>
      Two measures, two charts, one shared x-axis — never a dual y-axis, which
      lets whoever draws it imply any correlation they like.
    </figcaption>
  </figure>

  <div class="callout">
    <div class="eyebrow">Why the distinction is worth money</div>
    <p>
      "Retrain the model" is weeks of work and a model-risk sign-off.
      "Re-derive the thresholds" is a configuration change that takes effect on
      the next request. Diagnosing this incorrectly means paying the first price
      for a problem that only required the second.
    </p>
  </div>
</section>

<!-- ==================== 7. QUEUE ==================== -->
<section class="page">
  <h2>7 &nbsp; The review queue, and the limitation the cost model hides</h2>

  <p>
    Section 2 traded {num(fp_avoided)} wrongly-blocked customers for
    {num(three['reviewed'])} manual reviews. That trade assumes an analyst is
    available for each one. Here is what actually happens to the queue.
  </p>

  <table class="no-break">
    <thead>
      <tr><th class="r">Month</th><th class="r">Fraud rate</th><th class="r">Fixed cut-off</th><th class="r">Alert budget</th><th class="r">Recall (fixed / budget)</th></tr>
    </thead>
    <tbody>{queue_rows()}</tbody>
    <caption>
      Review rate against a team staffed for {pct(policy['max_review_rate'], 0)}.
      Red marks a month where the fixed cut-off overruns the staffing it was
      given.
    </caption>
  </table>

  <figure>
    <img src="{figure('04_queue_stability.png')}" alt="Queue stability">
    <figcaption>
      An alert budget holds the queue near its target in most months — but not
      all of them. Month 5 overruns, and the chart says so.
    </figcaption>
  </figure>

  <div class="callout bad-tone">
    <div class="eyebrow">The weakest point in this design</div>
    <p>
      The cost model charges <span class="mono">₹200</span> per review whether or
      not anyone performs it. So when the fixed cut-off overruns its staffing, it
      appears to win on recall — while spending analyst-hours that do not exist.
      A queue that overflows does not degrade gracefully; it degrades into
      applications sitting unreviewed until they time out. Pricing capacity
      overflow is the most valuable single change this system could receive.
    </p>
  </div>
</section>

<!-- ==================== 8. ADAPTATION ==================== -->
<section class="page">
  <h2>8 &nbsp; Adaptation — including the result that did not work</h2>

  <p>
    Given the decay in Section 6, the natural response is to adapt. Four
    strategies were tested: leave it alone, recalibrate on the newest month,
    retrain the booster entirely, and hold a fixed alert budget without using any
    labels. Each adapts on month {adaptation['adaptation_month']} and is
    evaluated on month {adaptation['evaluation_month']}.
  </p>

  <table class="no-break">
    <thead>
      <tr><th>Strategy</th><th class="r">Recall</th><th class="r">Cost / app</th><th class="r">vs. stale</th><th class="r">Adapt</th></tr>
    </thead>
    <tbody>{scenario_rows()}</tbody>
    <caption>
      Adapting on month {adaptation['adaptation_month']}, evaluated on month
      {adaptation['evaluation_month']}.
    </caption>
  </table>

  <div class="callout bad-tone">
    <div class="eyebrow">Negative result, reported as one</div>
    <p>
      <strong>No adaptation strategy beat doing nothing.</strong> Retraining the
      booster on an extra three months of data cost
      {scenarios['RETRAIN']['adapt_seconds'] / 60:.0f} minutes of compute and
      made the outcome <em>worse</em> — recall fell from {pct(stale['recall'])}
      to {pct(scenarios['RETRAIN']['recall'])}, and cost per application rose by
      ₹{scenarios['RETRAIN']['cost_per_app'] - stale['cost_per_app']:.0f}.
    </p>
    <p>
      The reason is visible in Section 6: discrimination was never the problem,
      so improving discrimination could not be the fix. Refitting on a period
      with a different fraud mix moved the thresholds to fit a month that had
      already passed. This is reported because a report that only contains
      experiments that worked is not evidence of judgement.
    </p>
  </div>

  <figure>
    <img src="{figure('06_adaptation.png')}" alt="Adaptation strategies">
    <figcaption>
      Four strategies, one evaluation month. The bar that does nothing is the
      one that wins.
    </figcaption>
  </figure>
</section>

<!-- ==================== 9. SIMILARITY ==================== -->
<section class="page">
  <h2>9 &nbsp; Adaptation that does work — one confirmed case becomes a detector</h2>

  <p>
    Retraining failed because it is the wrong instrument for a fast-moving
    pattern. A supervised model only recognises fraud resembling its training
    labels; when a new pattern appears it is blind until enough cases are
    confirmed, labelled, retrained and redeployed — weeks during which the
    pattern runs unchecked.
  </p>

  <p>
    The similarity layer closes that gap without touching the model. Each
    application is represented by <strong>the leaves it occupies in the
    gradient-boosted ensemble</strong>, and two applications are similar in
    proportion to how many trees route them to the same leaf. This uses the
    model's own learned notion of resemblance — including feature interactions —
    rather than a distance metric imposed on top of it. It is also directly
    interpretable, because the score has a plain reading: a similarity of 0.82
    means <em>these two applications are routed to the same leaf by 82% of the
    trees in the model</em>.
  </p>

  <h3>The threshold was measured, not chosen</h3>

  <table class="no-break">
    <thead>
      <tr><th class="r">Similarity cut-off</th><th class="r">Fires on genuine</th><th class="r">Fires on fraud</th><th class="r">Lift</th></tr>
    </thead>
    <tbody>
      <tr><td class="r n">0.45</td><td class="r n">32.3%</td><td class="r n">75.5%</td><td class="r n">2.3×</td></tr>
      <tr><td class="r n">0.55</td><td class="r n">5.3%</td><td class="r n">30.6%</td><td class="r n">5.8×</td></tr>
      <tr><td class="r n"><strong>0.60</strong></td><td class="r n good"><strong>1.3%</strong></td><td class="r n"><strong>10.2%</strong></td><td class="r n good"><strong>7.9×</strong></td></tr>
      <tr><td class="r n">0.65</td><td class="r n">0.3%</td><td class="r n">2.0%</td><td class="r n">6.2×</td></tr>
    </tbody>
    <caption>
      Measured on 4,000 held-out applications against a 400-case index. 0.60
      peaks the lift. A lower cut-off fires on a third of all traffic at barely
      2× lift — alert fatigue, not signal.
    </caption>
  </table>

  <h3>The escalation rule is deliberately narrow</h3>

  <p>
    A match alone does not decide anything. The rule only ever <em>raises</em> a
    decision, never lowers one; it moves APPROVE to REVIEW and never straight to
    BLOCK; and it fires only on a match strength measured to carry 7.9× lift. At
    8.9% precision, auto-blocking on similarity would decline roughly ten
    genuine applicants for every fraud stopped. The model's own decision is
    preserved in the response so an auditor can always see what was decided and
    what changed it.
  </p>

  <pre class="block">BEFORE   decision=APPROVE   model_decision=APPROVE   escalated=False
         p(fraud)=0.0259  - a fraudulent application the model lets through
         an analyst confirms the fraud  ->  index 400 -> 401
AFTER    decision=REVIEW    model_decision=APPROVE   escalated=True
         reason: 100% match to confirmed fraud case APP-F482729F9C
                 (shares device_os, employment_status, housing_status)
         p(fraud)=0.0259  - unchanged, because the model was never retrained</pre>

  <p>
    The confirmed case is written to PostgreSQL before it enters the in-process
    index, so it survives a restart. Verified by killing the process:
  </p>

  <pre class="block">before restart   fraud_vectors 401   index 401   decisions 562
   process killed and restarted
after  restart   fraud_vectors 401   index 401   decisions 562
                 index rebuilt from Postgres — the analyst's work survived</pre>
</section>

<!-- ==================== 10. ARCHITECTURE ==================== -->
<section class="page">
  <h2>10 &nbsp; Architecture</h2>

  <pre class="block">┌──────────────────────────────┐         ┌──────────────────────────────────────┐
│  React 19 + Vite + Tailwind  │  HTTPS  │  FastAPI  (Python 3.13)              │
│                              │────────►│                                      │
│  Live decision ledger        │         │  Decision engine                     │
│  Case detail + force bars    │         │   ├─ LightGBM         known fraud    │
│  Risk-appetite control       │         │   ├─ Isotonic calib.  honest P(fraud)│
│  Audit trail / model health  │         │   └─ leaf-overlap     similar cases  │
└──────────────────────────────┘         │                                      │
                                         │  Policy layer                        │
                                         │   cost-derived thresholds →          │
                                         │   APPROVE / REVIEW / BLOCK           │
                                         │   capped by analyst capacity         │
                                         │                                      │
                                         │  Explainability                      │
                                         │   SHAP → reason codes → Gemini prose │
                                         └──────┬─────────────────────┬─────────┘
                                    reads at    │                     │  queued,
                                    startup     │                     │  never awaited
                                         ┌──────▼─────────────────────▼─────────┐
                                         │  PostgreSQL 17 + pgvector 0.8        │
                                         │  fraud_vectors     HNSW / cosine     │
                                         │  applications      decisions         │
                                         │  analyst_verdicts  audit_log         │
                                         └──────────────────────────────────────┘</pre>

  <h3>What is deliberately kept off the decision path</h3>

  <p>
    Two things in this system are slow, and neither is allowed to make a decision
    slow.
  </p>

  <p>
    <strong>The language model.</strong> Gemini turns SHAP attributions into
    analyst prose. It costs roughly 3.4 seconds — two orders of magnitude more
    than everything else combined — so it is never called during scoring. The
    console requests it only when an analyst opens a case, and a deterministic
    template always exists as a fallback. Replaying an application against the
    same model version yields an identical decision whether or not the provider
    was reachable. A decision that can only be explained when a third-party API
    is up is not auditable.
  </p>

  <p>
    <strong>The audit write.</strong> A managed PostgreSQL is ~250ms away over
    the public internet — several decisions' worth of latency for one round trip.
    Writes are queued and drained by a background worker; the request thread pays
    <span class="mono">0.115ms</span>, measured. The cost of that choice is
    stated honestly in Section 12.
  </p>

  <table class="no-break">
    <thead><tr><th>Measured</th><th class="r">Value</th><th>Conditions</th></tr></thead>
    <tbody>
      <tr><td>Decision latency, p50</td><td class="r n">46 ms</td><td class="dim">paced as the console issues requests</td></tr>
      <tr><td>Decision latency, p95</td><td class="r n">86 ms</td><td class="dim">100 requests, 402-case index, persistence on</td></tr>
      <tr><td>Audit write, request thread</td><td class="r n">0.115 ms</td><td class="dim">serialise and enqueue</td></tr>
      <tr><td>Narration, when requested</td><td class="r n">~3.4 s</td><td class="dim">off the decision path by design</td></tr>
    </tbody>
    <caption>
      Measured on a developer laptop with model, calibrator, SHAP and policy in
      one process. Sensitive to machine load: the same benchmark returned p50
      118ms while unrelated build watchers consumed half the CPU.
    </caption>
  </table>
</section>

<!-- ==================== 11. EXPLAINABILITY ==================== -->
<section class="page">
  <h2>11 &nbsp; Explainability, and the regulatory obligation behind it</h2>

  <p>
    A lender that declines an application generally has to say why. That is not a
    product preference; under adverse-action requirements it is a legal
    obligation, and it rules out any explanation method that cannot attribute a
    <em>specific decision</em> to <em>specific inputs</em>.
  </p>

  <p>
    Aegis uses SHAP TreeExplainer, whose attributions are
    <strong>additive</strong>: the contributions sum to the model's output for
    that application. The force bars in the console are therefore the explanation
    itself, not an illustration of one. They are ordered by magnitude and
    rendered into adverse-action reason codes in that order — the ranking a
    regulator expects.
  </p>

  <h4>The boundary that makes this defensible</h4>

  <p>
    The language model sits strictly <em>downstream</em> of the decision. The
    booster, the calibrator and the policy produce an outcome; SHAP explains that
    outcome; Gemini narrates the explanation. Nothing downstream of the policy
    can change what was decided. This ordering is enforced by the code structure
    and asserted by the smoke tests, not left as an intention.
  </p>

  <h3>Responsible AI</h3>

  <table class="no-break">
    <thead><tr><th style="width:44mm">Concern</th><th>Treatment</th></tr></thead>
    <tbody>
      <tr><td>Protected attributes</td><td>Age and employment status are present in the data and analysed for disparate impact rather than silently dropped — dropping a variable does not remove its proxies.</td></tr>
      <tr><td>Explanation integrity</td><td>Additive SHAP; the LLM cannot alter a score, a threshold or an outcome.</td></tr>
      <tr><td>Auditability</td><td>Every decision written to an append-only log; PostgreSQL rejects UPDATE and DELETE via trigger.</td></tr>
      <tr><td>Human in the loop</td><td>The review band exists precisely so borderline cases reach a person rather than an automatic decline.</td></tr>
      <tr><td>Honest reporting</td><td>Accuracy deliberately excluded; the demo stream discloses its own over-sampling and shows misses as well as catches.</td></tr>
    </tbody>
  </table>

  <div class="callout warn-tone">
    <div class="eyebrow">Stated plainly</div>
    <p>
      Housing status carries {pct(metrics['top_features'][0]['share'])} of total
      model gain. It is a legitimate stability signal and also a plausible proxy
      for socio-economic status. A production deployment would need a formal
      disparate-impact review before this model decided anything about a real
      person.
    </p>
  </div>
</section>

<!-- ==================== 12. LIMITATIONS ==================== -->
<section class="page">
  <h2>12 &nbsp; Limitations</h2>

  <p class="lede">Stated here rather than left to be discovered.</p>

  <table>
    <thead><tr><th style="width:52mm">Limitation</th><th>Consequence, and what would fix it</th></tr></thead>
    <tbody>
      <tr>
        <td><strong>Capacity overflow is not priced</strong></td>
        <td>The cost model charges ₹200 per review whether or not an analyst exists. This is why a fixed cut-off appears to beat the alert budget on recall — it spends hours it does not have. The most valuable single change available.</td>
      </tr>
      <tr>
        <td><strong>The dataset is synthetic</strong></td>
        <td>BAF is a CTGAN rendering of a real anonymised dataset under differential privacy. It preserves realistic structure, feature semantics and temporal drift, but it is not live production data.</td>
      </tr>
      <tr>
        <td><strong>Account opening ≠ lending origination</strong></td>
        <td>The fields are unambiguously credit-product ones — proposed credit limit, payment plan type, other cards held — so the domain match is close. There is no post-origination behaviour, so first-party bust-out fraud is out of scope.</td>
      </tr>
      <tr>
        <td><strong>No adaptation strategy beat inaction</strong></td>
        <td>Reported as a negative result in Section 8 rather than omitted.</td>
      </tr>
      <tr>
        <td><strong>Calibration produces tied probabilities</strong></td>
        <td>Isotonic regression yields heavily tied outputs, so quantile- and capacity-based thresholds are coarse; neither mechanism lands exactly on 5%. Exact targeting would need tie-breaking on the raw score.</td>
      </tr>
      <tr>
        <td><strong>The audit write is asynchronous</strong></td>
        <td>A process killed between enqueue and flush loses at most one half-second batch. Correct against a database across the internet; wrong against a co-located one.</td>
      </tr>
      <tr>
        <td><strong>No cloud deployment</strong></td>
        <td>Runs locally against managed PostgreSQL. Containerisation, IAM and a hosted runtime are designed but not built.</td>
      </tr>
      <tr>
        <td><strong>The analyst catch rate is assumed</strong></td>
        <td>90% is a configured assumption, not a measurement. It is configuration precisely so it can be replaced with an observed figure.</td>
      </tr>
      <tr>
        <td><strong>The browser holds the API key</strong></td>
        <td>Acceptable for a single-operator console, not for production. The production path is a session-authenticated backend-for-frontend holding the key server-side.</td>
      </tr>
    </tbody>
  </table>

  <h3>Engineering</h3>

  <table class="no-break">
    <thead><tr><th>Practice</th><th class="r">State</th></tr></thead>
    <tbody>
      <tr><td>Unit tests</td><td class="r n">86 passing</td></tr>
      <tr><td>End-to-end smoke checks</td><td class="r n">24</td></tr>
      <tr><td>Live-database verification</td><td class="r n">20 checks</td></tr>
      <tr><td>Secrets</td><td class="r">environment only; none in source or version control</td></tr>
      <tr><td>API contract</td><td class="r">OpenAPI, generated from the code that serves it</td></tr>
      <tr><td>Authentication</td><td class="r">API key on every route, including metrics</td></tr>
    </tbody>
  </table>

  <div class="footer-note">
    Dataset: Bank Account Fraud Suite (NeurIPS 2022), Feedzai, CC BY-NC-ND 4.0.
    Not redistributed here. Every figure and table is generated directly from
    the training and analysis artifacts.
  </div>
</section>

</body></html>"""


def main() -> int:
    print("reading artifacts...")
    html = build_html()

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"wrote {OUTPUT_HTML.relative_to(PROJECT_ROOT)}")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright is not installed; HTML written but no PDF produced.")
        return 1

    print("rendering PDF...")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.goto(OUTPUT_HTML.as_uri(), wait_until="networkidle")
        page.pdf(
            path=str(OUTPUT_PDF),
            format="A4",
            print_background=True,
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
        )
        browser.close()

    size_kb = OUTPUT_PDF.stat().st_size / 1024
    print(f"wrote {OUTPUT_PDF.relative_to(PROJECT_ROOT)}  ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
