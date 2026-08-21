"""Generate the console walkthrough as a PDF.

Preparation material, not a submission artifact. It explains every panel of the
operations console in the order they were talked through, so the whole interface
can be narrated fluently without notes.

It deliberately keeps the things a neutral document would strip: what to point
at, what looks bad and why it is fine, and the questions each panel invites.
That is the entire reason it exists, and it is why the cover says plainly that
it does not belong in the submission zip.

Screenshots come from `docs/figures/guide`, captured against the running
console. Policy numbers are read from the training artifacts so the arithmetic
in the text cannot drift from the deployed model.

Usage:
    python docs/build_console_guide.py
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS.parent
ARTIFACTS = PROJECT_ROOT / "backend" / "ml" / "artifacts"
GUIDE_FIGURES = DOCS / "figures" / "guide"

OUTPUT_PDF = DOCS / "Aegis-Console-Guide-PREP.pdf"
OUTPUT_HTML = DOCS / "console_guide.html"

# Share the report's typography rather than inventing a second visual language
# for the same project.
sys.path.insert(0, str(DOCS))
from build_report import STYLES as BASE_STYLES  # noqa: E402


def load(name: str) -> dict:
    path = ARTIFACTS / name
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run the training scripts first.")
    return json.loads(path.read_text(encoding="utf-8"))


def shot(name: str) -> str:
    path = GUIDE_FIGURES / name
    if not path.exists():
        print(f"  warning: missing screenshot {name}", file=sys.stderr)
        return ""
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def pct(value: float, places: int = 1) -> str:
    return f"{value * 100:.{places}f}%"


EXTRA_STYLES = """
/* ---------- prep banner ---------- */
.prep-banner {
  border: 2px solid var(--block); background: #fdf3f1;
  padding: 4mm 5mm; margin: 6mm 0 0;
}
.prep-banner .eyebrow { color: var(--block); }
.prep-banner p { margin: 1.5mm 0 0; font-size: 9.6pt; }

/* ---------- screenshots ---------- */
.shot {
  border: 1px solid var(--rule-2); background: #0a0a0a;
  padding: 0; margin: 0; line-height: 0;
}
.shot img { width: 100%; display: block; }

.shot-wide { margin: 3mm 0 2mm; }
.shot-wide img { width: 100%; display: block; border: 1px solid var(--rule-2); }

/* Tall panel beside its notes. */
.panel-split {
  display: grid; grid-template-columns: 62mm 1fr; gap: 0 7mm;
  align-items: start; margin: 4mm 0;
}
.panel-split .shot img { max-height: 190mm; object-fit: contain; object-position: top; }
.panel-split .notes p:first-child { margin-top: 0; }

.figure-note {
  font-size: 8pt; color: var(--ink-3); line-height: 1.4;
  margin: 1.5mm 0 0;
}

/* ---------- say-this blocks ---------- */
.say {
  border-left: 2.5px solid var(--approve); background: #f2f6f1;
  padding: 3mm 4.5mm; margin: 3.5mm 0; page-break-inside: avoid;
}
.say .eyebrow { color: var(--approve); margin-bottom: 1.5mm; }
.say p { margin: 0; font-style: italic; font-size: 9.8pt; }

/* ---------- expected-question blocks ---------- */
.qa { margin: 3mm 0 0; page-break-inside: avoid; }
.qa dt {
  font-family: "Segoe UI", system-ui, sans-serif; font-weight: 600;
  font-size: 9.4pt; margin-top: 3mm;
}
.qa dd { margin: 1mm 0 0; font-size: 9.6pt; }

.anatomy { width: 100%; font-size: 9pt; }
.anatomy td { padding: 1.4mm 3mm 1.4mm 0; border-bottom: 1px solid var(--rule); }
.anatomy td:first-child {
  font-family: "Segoe UI", system-ui, sans-serif;
  font-weight: 600; width: 46mm; vertical-align: top;
}
"""


def build_html() -> str:
    metrics = load("metrics.json")
    policy = metrics["policy"]

    c_fp, c_fn = 1_500.0, 45_000.0
    c_rev, r = 200.0, 0.90
    tau_review_raw = c_rev / (r * c_fn)
    tau_block = (c_fp - c_rev) / (c_fn * (1 - r) + c_fp)

    # The observed case used throughout section 1-3. Captured live; see
    # scratchpad/example_case.py in the session that produced it.
    p_case = 0.09091

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Aegis — Console Guide (preparation notes)</title>
<style>{BASE_STYLES}{EXTRA_STYLES}</style></head><body>

<!-- ==================== COVER ==================== -->
<section class="page cover">
  <div class="cover-top">
    <div class="eyebrow">Preparation notes</div>
    <h1>Console<span class="accent">A walkthrough of every panel, and what
      to say about it</span></h1>
    <div class="rule-heavy"></div>
    <p class="sub">
      The operations console makes five distinct arguments, one per region of
      the screen. This explains each of them, what the numbers mean, where they
      look worse than they are, and the questions each panel invites.
    </p>

    <div class="prep-banner">
      <div class="eyebrow">Not part of the submission</div>
      <p>
        This document contains rehearsal notes — what to point at, what to say,
        and how to answer challenges. It is for preparation only and should not
        go in the submission archive. The technical report
        (<span class="mono">Aegis-Technical-Report.pdf</span>) is the document
        intended for readers.
      </p>
    </div>

    <div class="shot-wide" style="margin-top:7mm">
      <img src="{shot('console_full.png')}" alt="The full console">
    </div>
    <p class="figure-note">
      Three columns, left to right, in the order the work happens: what arrived,
      why it was decided, the policy that decided it. The metrics strip runs
      across the top; the audit trail shares the lower-right panel with model
      health.
    </p>
  </div>
</section>

<!-- ==================== 1. RISK APPETITE ==================== -->
<section class="page">
  <h2>1 &nbsp; Risk appetite</h2>

  <p class="lede">
    Four business costs go in; two decision thresholds come out. That is the
    whole panel.
  </p>

  <p>
    The argument it exists to make: <strong>the approve/review/block boundaries
    are not model settings.</strong> They are not tuned, not searched for, not
    learned. They are arithmetic over what errors cost the lender — which means
    a risk owner can change the bank's appetite without a data scientist, a
    retrain, or a model-risk sign-off.
  </p>

  <div class="panel-split">
    <div class="shot">
      <img src="{shot('risk_appetite.png')}" alt="Risk appetite panel">
    </div>
    <div class="notes">
      <h4>The four inputs</h4>
      <table class="anatomy">
        <tr><td>Cost of blocking a genuine customer<br><span class="num dim">₹{c_fp:,.0f}</span></td>
            <td>The false positive. Lost acquisition, servicing, reputational drag.</td></tr>
        <tr><td>Cost of approving a fraud<br><span class="num dim">₹{c_fn:,.0f}</span></td>
            <td>The false negative. Expected write-off on the credit extended.</td></tr>
        <tr><td>Cost of one manual review<br><span class="num dim">₹{c_rev:,.0f}</span></td>
            <td>Analyst time. Must stay below the cost of a wrongful block, or
                reviewing is never worth doing.</td></tr>
        <tr><td>Analyst catch rate<br><span class="num dim">{r:.0%}</span></td>
            <td>Probability an analyst correctly identifies fraud once the case
                reaches them.</td></tr>
      </table>
    </div>
  </div>

  <h3>How the thresholds are derived</h3>

  <p>
    At any fraud probability <span class="mono">p</span>, each of the three
    possible actions has an expected cost. Each is a straight line in
    <span class="mono">p</span>; the optimal action is whichever line is lowest,
    and the thresholds are simply where the lines cross.
  </p>

  <div class="formula">
APPROVE   p · C<sub>fn</sub>                      lose the principal if it is fraud
BLOCK     (1 − p) · C<sub>fp</sub>                lose a customer if it is genuine
REVIEW    C<sub>rev</sub> + p · (1 − r) · C<sub>fn</sub>    pay the analyst; lose only what they miss

τ<sub>review</sub> = C<sub>rev</sub> / (r · C<sub>fn</sub>)                       = {tau_review_raw:.5f}
τ<sub>block</sub>  = (C<sub>fp</sub> − C<sub>rev</sub>) / (C<sub>fn</sub>(1 − r) + C<sub>fp</sub>)     = {tau_block:.5f}
  </div>

  <p>Nothing was searched. No validation set was consulted.</p>
</section>

<!-- ==================== 1b. RISK APPETITE, WORKED ==================== -->
<section class="page">
  <h2>1 &nbsp; Risk appetite <span class="dim" style="font-weight:400">— worked through</span></h2>

  <h3>Watch it work on a real case</h3>

  <p>
    <span class="mono">APP-31120547AC</span> scored
    <strong>p = {pct(p_case, 2)}</strong>. The ground truth for that
    application is <strong>fraud</strong>.
  </p>

  <table class="no-break">
    <thead><tr><th>Action</th><th>Expected cost at p = {pct(p_case, 2)}</th><th class="r">Cost</th></tr></thead>
    <tbody>
      <tr><td>Approve</td><td class="mono dim">{p_case:.5f} × {c_fn:,.0f}</td>
          <td class="r n">₹{p_case * c_fn:,.0f}</td></tr>
      <tr><td>Block</td><td class="mono dim">{1 - p_case:.5f} × {c_fp:,.0f}</td>
          <td class="r n">₹{(1 - p_case) * c_fp:,.0f}</td></tr>
      <tr class="total"><td><strong>Review</strong></td>
          <td class="mono dim">{c_rev:.0f} + {p_case:.5f} × {1 - r:.2f} × {c_fn:,.0f}</td>
          <td class="r n good">₹{c_rev + p_case * (1 - r) * c_fn:,.0f}</td></tr>
    </tbody>
    <caption>
      The system said REVIEW — not because 9% "feels borderline", but because
      reviewing is the cheapest of three priced options. It was fraud, so the
      review band earned its keep on this case.
    </caption>
  </table>

  <h3>The one number that is not pure arithmetic</h3>

  <p>
    The panel derives <span class="mono">τ<sub>review</sub> =
    {tau_review_raw:.5f}</span>, but the live threshold is
    <strong>{policy['tau_review']:.5f}</strong> — nine times higher. That is the
    <strong>analyst-capacity cap</strong>.
  </p>

  <p>
    The Bayes-optimal threshold assumes unlimited review capacity. At
    {pct(tau_review_raw, 2)} it would route an enormous share of traffic to
    humans. A second constraint caps the review band at the volume the team can
    actually work ({pct(policy['max_review_rate'], 0)}), which raises the
    threshold. It is applied <em>after</em> the economics, deliberately, so the
    trade stays visible instead of being buried in a tuned constant.
  </p>

  <p>
    It is also why the panel's own footnote says live thresholds are
    additionally capped: the cap needs a live score distribution, so it happens
    at serving time rather than in the simulator.
  </p>

  <div class="say">
    <div class="eyebrow">The demo moment — drag the first slider upward</div>
    <p>
      "I have not retrained anything. These thresholds are derived from four
      business costs by minimising expected loss. Raising the cost of a wrongful
      block makes the system measurably less willing to block. Changing the
      bank's risk appetite is arithmetic, not a model release."
    </p>
  </div>

  <p>
    The recompute is server-side, calling the same
    <span class="mono">derive_policy()</span> the scoring path uses — so what is
    on screen is the real derivation, not a JavaScript re-implementation that
    could drift from it.
  </p>

  <dl class="qa">
    <dt>"Where did ₹45,000 come from?"</dt>
    <dd>It is configuration, not a measurement, and that is the point — it lives
      in <span class="mono">.env</span> so a risk owner can set it. A real
      deployment would use the observed average write-off.</dd>
    <dt>"Why cap review at 5%?"</dt>
    <dd>Because a policy that is optimal on paper and unstaffable in practice is
      not optimal. It is the difference between a threshold derived in a
      notebook and one deployable on Monday.</dd>
  </dl>
</section>

<!-- ==================== 2. CASE DETAIL ==================== -->
<section class="page">
  <h2>2 &nbsp; Case detail</h2>

  <p class="lede">
    Ordered the way an analyst actually reads a case: what was decided, why,
    what it resembles, and what to do.
  </p>

  <div class="panel-split">
    <div class="shot">
      <img src="{shot('case_detail.png')}" alt="Case detail panel">
    </div>
    <div class="notes">
      <table class="anatomy">
        <tr><td>Header</td><td>Application ID, decision pill, and the
          <em>calibrated</em> fraud probability. "Calibrated" matters — an
          uncalibrated score makes every cost comparison in Section 1
          invalid.</td></tr>
        <tr><td>Threshold meter</td><td>Where this application landed relative
          to the two live thresholds. The scale is square-root, not linear:
          review sits at {policy['tau_review']:.3f} and block at
          {policy['tau_block']:.3f}, so on a linear axis every marker would pile
          against the left edge and show nothing.</td></tr>
        <tr><td>Rule override</td><td>Appears only when the similarity layer
          escalated the decision. Shows the model's own verdict alongside the
          final one.</td></tr>
        <tr><td>Ground truth</td><td>Replay only. The dataset's true label, so
          misses are visible as well as catches. The scoring path never sees
          this.</td></tr>
        <tr><td>Analyst briefing</td><td>Plain-English summary of the
          attributions, tagged <span class="mono">GEMINI</span> or
          <span class="mono">TEMPLATE</span>.</td></tr>
        <tr><td>Principal reasons</td><td>Adverse-action codes. Shown only for
          REVIEW and BLOCK.</td></tr>
        <tr><td>Factor contributions</td><td>The SHAP force bars — Section 3.</td></tr>
        <tr><td>Resembles confirmed fraud</td><td>Nearest cases in the vector
          index, with a strength label.</td></tr>
        <tr><td>Record verdict</td><td>Confirm fraud / clear as genuine. This is
          the feedback loop.</td></tr>
      </table>
    </div>
  </div>

  <p class="figure-note">
    The case above is a REVIEW at 5.7% whose ground truth is <em>genuine</em> —
    a needless review. Worth showing rather than hiding: the review band's cost
    is real, and it is paid on cases like this one.
  </p>
</section>

<!-- ==================== 2b. THE BRIEFING ==================== -->
<section class="page">
  <h2>2 &nbsp; Case detail <span class="dim" style="font-weight:400">— the analyst briefing</span></h2>

  <p>
    A plain-English paragraph summarising the attributions, written
    <strong>after</strong> the decision, from numbers that already exist:
  </p>

  <pre class="block">Decision: REVIEW. The strongest driver was match between applicant name
and email address (0.12). Also contributing: internal credit risk score
(267); session kept alive during application (0). Offsetting this, device
operating system (linux) reduced the assessed risk.</pre>

  <p>
    The tag in the corner reads <span class="mono">GEMINI</span> or
    <span class="mono">TEMPLATE</span>. During rehearsal it will often say
    TEMPLATE, because the free-tier quota is exhausted and the deterministic
    fallback is doing the work.
  </p>

  <div class="callout">
    <div class="eyebrow">The most important boundary in the system</div>
    <p>
      Gemini receives the computed SHAP values and turns them into prose. It
      <strong>cannot alter a score, a threshold, or an outcome.</strong> Two
      consequences worth saying out loud:
    </p>
    <p style="margin-top:2mm">
      It is off the <em>decision</em> path — narration costs ~3.4s against an
      86ms decision, so the stream never waits for it; it is fetched only when a
      case is opened. And it is off the <em>correctness</em> path — a decision
      that can only be explained when a third-party API is reachable is not
      auditable, so the template always exists.
    </p>
  </div>

  <h3>The briefing and the reason codes are different things</h3>

  <p>Worth not conflating, because a sharp question will separate them.</p>

  <table class="no-break">
    <thead><tr><th></th><th>Audience</th><th>Source</th></tr></thead>
    <tbody>
      <tr><td><strong>Analyst briefing</strong></td><td>The analyst working the case</td>
          <td>LLM prose, or a deterministic template</td></tr>
      <tr><td><strong>Principal reasons</strong></td><td>The declined customer</td>
          <td>The SHAP ordering, verbatim</td></tr>
    </tbody>
  </table>

  <p>
    Under Regulation B a declined applicant is entitled to the <em>specific
    principal reasons</em>. Those come straight from the top-four risk factors by
    attribution magnitude — never paraphrased, never LLM-generated. That is what
    makes them genuinely principal rather than a generic list.
  </p>

  <div class="say">
    <div class="eyebrow">If asked why the LLM is not making decisions</div>
    <p>
      "Because then the decision would not be reproducible, and a lending
      decision that cannot be reconstructed months later is not defensible. The
      model, the calibrator and the policy decide; SHAP explains; Gemini writes.
      Nothing downstream of the policy can change what was decided."
    </p>
  </div>
</section>

<!-- ==================== 3. FACTORS ==================== -->
<section class="page">
  <h2>3 &nbsp; Reading the factor contributions</h2>

  <p>
    SHAP TreeExplainer, computed per application — not global feature
    importance, which describes the model in general and says nothing about
    <em>this</em> applicant. The values are <strong>additive in
    log-odds</strong>:
  </p>

  <div class="formula">
base_value  +  Σ(all SHAP values)  =  the model's raw score for this application
  </div>

  <p>
    That additivity is the point. The bars do not illustrate the explanation —
    they <strong>are</strong> the explanation, and they sum to the decision.
    Bars extending right raise risk; left lowers it.
  </p>

  <p>
    One subtlety worth having ready: attributions are computed against the
    booster's <em>raw</em> output, not the calibrated probability. Isotonic
    calibration is monotonic, so it cannot reorder anything — explaining the raw
    score keeps the attributions exactly additive.
  </p>

  <h3>The live case, factor by factor</h3>

  <p><span class="mono">APP-31120547AC</span> — REVIEW at {pct(p_case, 2)}, actually fraud.</p>

  <div class="two-col">
    <div>
      <h4>Raising risk</h4>
      <table class="no-break" style="font-size:8.6pt">
        <tbody>
          <tr><td class="n bad">+1.1672</td><td>Name/email match <span class="dim">0.13</span></td></tr>
          <tr><td class="n bad">+0.9096</td><td>Device OS <span class="dim">windows</span></td></tr>
          <tr><td class="n bad">+0.4924</td><td>Time at previous address <span class="dim">not provided</span></td></tr>
          <tr><td class="n bad">+0.4908</td><td>Time at current address <span class="dim">60</span></td></tr>
          <tr><td class="n bad">+0.4603</td><td>Emails sharing this DOB <span class="dim">3</span></td></tr>
        </tbody>
      </table>
    </div>
    <div>
      <h4>Lowering risk</h4>
      <table class="no-break" style="font-size:8.6pt">
        <tbody>
          <tr><td class="n good">−0.4451</td><td>Housing status <span class="dim">BE</span></td></tr>
          <tr><td class="n good">−0.3179</td><td>Declared income band <span class="dim">0.10</span></td></tr>
          <tr><td class="n good">−0.2986</td><td>Banking relationship age <span class="dim">1.00</span></td></tr>
          <tr><td class="n good">−0.2425</td><td>Application rate, 24h <span class="dim">2,834</span></td></tr>
          <tr><td class="n good">−0.2062</td><td>Time since request started <span class="dim">0.02</span></td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <p>
    The top factor scores <strong>0.13 on name–email similarity</strong> — the
    applicant's name barely matches their email. That alone contributes +1.17
    log-odds, more than the next two combined. Three distinct emails share this
    date of birth, and no previous address was given: a coherent
    identity-fabrication signature, and it was in fact fraud.
  </p>

  <div class="callout warn-tone">
    <div class="eyebrow">Two caveats to have ready</div>
    <p>
      <strong>The feature values look strange because BAF ships them
      pre-engineered.</strong> <span class="mono">income = 0.10</span> is a
      normalised band, not rupees. <span class="mono">name_email_similarity =
      0.13</span> is a 0–1 score. Expect to be asked.
    </p>
    <p style="margin-top:2mm">
      <strong>SHAP shows what the model learned, not what causes fraud.</strong>
      Note <span class="mono">Application rate in the last 24 hours</span> in the
      <em>protective</em> column. High velocity sounds suspicious, yet the model
      reads it as risk-reducing here — an interaction effect. It is a
      correlational attribution from a synthetic dataset, and exactly why a
      production deployment needs a disparate-impact review. Acknowledge it; do
      not explain it away.
    </p>
  </div>

  <h3>Where the similarity panel fits</h3>

  <p>
    Separate from SHAP entirely. Weak matches mean "nothing on file resembles
    this strongly enough to act on" — stated explicitly rather than shown as an
    empty box, because an analyst cannot distinguish <em>no resemblance
    found</em> from <em>this feature is broken</em>. Only a <strong>strong</strong>
    match (≥ 0.60, measured at 7.9× lift) triggers escalation, and even then only
    APPROVE → REVIEW, never straight to BLOCK.
  </p>
</section>

<!-- ==================== 4. LEDGER ==================== -->
<section class="page">
  <h2>4 &nbsp; The live decision ledger</h2>

  <p class="lede">
    Applications replayed through the real scoring endpoint, one at a time,
    newest first.
  </p>

  <p>
    Nothing here is simulated. Each row is a genuine HTTP call to
    <span class="mono">POST /api/v1/applications/score</span>, and the latency
    shown is what the server actually took.
  </p>

  <div class="panel-split">
    <div class="shot">
      <img src="{shot('ledger.png')}" alt="Live decision ledger">
    </div>
    <div class="notes">
      <table class="anatomy">
        <tr><td>Application</td><td>The ID, with a thin left rule in the decision
          colour — lets the eye group the column without a second badge.</td></tr>
        <tr><td>P(fraud)</td><td>Calibrated probability. Precision scales with
          magnitude (<span class="mono">58.33%</span>,
          <span class="mono">4.49%</span>, <span class="mono">&lt;0.01%</span>);
          a fixed decimal count would floor small values to "0%".</td></tr>
        <tr><td>Decision</td><td>Colour <em>plus</em> glyph <em>plus</em> text.
          Roughly 1 in 12 men has a colour vision deficiency; encoding the
          verdict in hue alone would make this unusable for them.</td></tr>
        <tr><td>Outcome</td><td>The demonstration affordance — see below.</td></tr>
        <tr><td>Latency</td><td>Server-side, for that specific decision.</td></tr>
      </table>

      <p class="figure-note" style="margin-top:3mm">
        Streams at 900ms — fast enough to feel live, slow enough to read.
        Bounded at 60 rows so a long demo cannot grow the DOM without limit, and
        pinned to the top so the newest decision never scrolls out of view.
      </p>
    </div>
  </div>

  <h3>The Outcome column is the honest bit</h3>

  <p>Because held-out data is being replayed, the true label is known.</p>

  <table class="no-break">
    <thead><tr><th></th><th>Actually fraud</th><th>Actually genuine</th></tr></thead>
    <tbody>
      <tr><td><strong>BLOCK</strong></td>
          <td class="good">caught</td><td class="bad">false positive</td></tr>
      <tr><td><strong>REVIEW</strong></td>
          <td class="warn">to analyst</td><td class="warn">needless review</td></tr>
      <tr><td><strong>APPROVE</strong></td>
          <td class="bad">MISSED</td><td class="dim">clean</td></tr>
    </tbody>
    <caption>
      Note the colour logic: BLOCK is red as a <em>decision</em> but
      <span class="good">caught</span> is green as an <em>outcome</em>. Severity
      and correctness are different axes, and conflating them would mislead.
    </caption>
  </table>

  <div class="say">
    <div class="eyebrow">Point at a MISSED row</div>
    <p>
      "That one was fraud and we approved it. The ledger shows misses, not just
      catches — a fraud console that only displays its wins is a sales pitch, and
      the first sharp question would dismantle it."
    </p>
  </div>
</section>

<!-- ==================== 5. METRICS STRIP ==================== -->
<section class="page">
  <h2>5 &nbsp; The metrics strip</h2>

  <div class="shot-wide">
    <img src="{shot('metrics_strip.png')}" alt="Metrics strip">
  </div>

  <p>
    Six tiles — and the non-obvious thing is that <strong>four are live session
    counters and two are fixed properties of the model.</strong> Worth knowing
    before someone asks why two of them never move.
  </p>

  <div class="two-col">
    <div>
      <h4>The four live counters</h4>
      <table class="anatomy">
        <tr><td>Scored</td><td>Rows this session</td></tr>
        <tr><td>Blocked</td><td>Count of BLOCK; hint shows the REVIEW count</td></tr>
        <tr><td>Fraud caught</td><td><span class="mono">caught / (caught + missed)</span></td></tr>
        <tr><td>Mean latency</td><td>Mean of <span class="mono">latency_ms</span></td></tr>
      </table>
    </div>
    <div>
      <h4>The two static ones</h4>
      <table class="anatomy">
        <tr><td>Behavioural signal<br><span class="num dim">{pct(metrics['behavioural_gain_share'], 0)}</span></td>
            <td>Share of total model gain from velocity, device and session
              features</td></tr>
        <tr><td>Calibration error<br><span class="num dim">{metrics['test_calibrated']['ece']:.4f}</span></td>
            <td>Held-out ECE, from {metrics['test_raw']['ece']:.3f} before
              isotonic calibration</td></tr>
      </table>
    </div>
  </div>

  <p>
    The static pair come from <span class="mono">metrics.json</span>, measured
    once on the test months. They are on screen because they are the two
    properties that most justify the design — not because they move.
  </p>

  <dl class="qa">
    <dt>"Why is accuracy not on the strip?"</dt>
    <dd>At 1.4% prevalence, predicting "never fraud" scores 98.6%. Leading with
      accuracy here reports the base rate. Its absence is deliberate, and is
      itself a point worth making.</dd>
  </dl>
</section>

<!-- ==================== 5b. METRICS CAVEATS ==================== -->
<section class="page">
  <h2>5 &nbsp; The metrics strip <span class="dim" style="font-weight:400">— three caveats</span></h2>

  <p class="lede">
    Each of these is a question waiting to be asked. Better to say it first.
  </p>

  <div class="callout warn-tone">
    <div class="eyebrow">1 — "Fraud caught" counts a REVIEW as a catch</div>
    <p>
      The arithmetic is <span class="mono">caught = actual fraud AND decision !=
      APPROVE</span>. Routing a fraud to an analyst counts as caught, even though
      it is only a real catch if the analyst catches it — assumed at
      {r:.0%}. The tile is <strong>optimistic by construction</strong>. Better to
      say so than to be caught by it.
    </p>
  </div>

  <div class="callout warn-tone">
    <div class="eyebrow">2 — The number will look bad early</div>
    <p>
      A real 25-application sample gave <strong>20% caught, 4 missed</strong>,
      against a headline of {pct(metrics['test_calibrated']['recall_at_5pct'])}
      recall at a 5% alert rate. The denominator was five frauds; one more catch
      swings it to 40%. Let the stream run. If the number is ugly on camera, say
      "five frauds is not a sample" rather than looking uncomfortable.
    </p>
  </div>

  <div class="callout warn-tone">
    <div class="eyebrow">3 — The stream is over-sampled, and discloses it</div>
    <p>
      The panel header reads <em>true fraud rate 1.40%</em> while the feed is
      stratified to <strong>20% fraud</strong>, one every five positions. At the
      natural rate a sixty-second demo would show roughly one fraudulent
      application, which demonstrates nothing.
    </p>
    <p style="margin-top:2mm">
      <strong>Fraud caught is still valid</strong> — recall is TP/(TP+FN), which
      does not depend on prevalence provided the fraud cases were randomly
      sampled. They were. <strong>Blocked counts are not</strong> — 1 block in 25
      is an artefact of 20% fraud, not a production block rate.
    </p>
  </div>

  <dl class="qa">
    <dt>"Are you inflating the numbers by over-sampling fraud?"</dt>
    <dd>Every stream response returns
      <span class="mono">sampled_fraud_share</span>,
      <span class="mono">true_fraud_rate</span> and a written disclosure string,
      and the console prints the true rate on screen. Stated, not hidden — and
      recall is invariant to it.</dd>
  </dl>
</section>

<!-- ==================== 6. AUDIT TRAIL ==================== -->
<section class="page">
  <h2>6 &nbsp; The audit trail <span class="dim" style="font-weight:400">— lower-right tab</span></h2>

  <p class="lede">
    Everything the system decides, written to PostgreSQL and impossible to
    rewrite.
  </p>

  <div class="panel-split">
    <div class="shot">
      <img src="{shot('audit_trail.png')}" alt="Audit trail panel">
    </div>
    <div class="notes">
      <p>
        Three counters, an append-only notice, and a live tail of decisions and
        analyst verdicts.
      </p>
      <table class="anatomy">
        <tr><td>Decisions</td><td>Rows in the <span class="mono">decisions</span>
          table</td></tr>
        <tr><td>Audit rows</td><td>Entries in
          <span class="mono">audit_log</span></td></tr>
        <tr><td>Fraud vectors</td><td>Confirmed-fraud embeddings in pgvector —
          400 seeded from history</td></tr>
      </table>
      <p>
        The header badge reads <span class="mono">postgres</span> in green when
        recording is live. It turns amber or red if the write queue backs up or
        drops anything — a green light over a queue that is not draining would be
        worse than no light at all.
      </p>
    </div>
  </div>

  <div class="say">
    <div class="eyebrow">The strongest beat in the demo — do not rush it</div>
    <p>
      "Every decision you just watched is in Postgres. The audit table is
      append-only, enforced by a database trigger rather than by convention —
      nothing in this application can rewrite a recorded decision. Now watch: I
      confirm one case as fraud, the index goes from 400 to 401, and the next
      application matching that pattern is escalated automatically. No
      retraining. And because it is in Postgres, it survives a restart."
    </p>
  </div>

  <p>
    The audit counters surviving a restart the process did not choose is the
    single clearest demonstration that persistence is real. If the servers get
    restarted during rehearsal, point at the row count afterwards.
  </p>

  <div class="callout bad-tone">
    <div class="eyebrow">Do not confirm a case during rehearsal</div>
    <p>
      Confirming a fraud writes a <em>permanent</em> vector. Doing it on an
      application that was actually genuine teaches the index that a genuine
      profile is fraudulent, and later lookalikes get escalated for nothing —
      which then shows up on camera as an unexplained
      <span class="mono">NEEDLESS REVIEW</span>. Clean up with:
    </p>
    <p style="margin-top:2mm" class="mono">
      python scripts/reset_demo.py --vectors-only --apply
    </p>
    <p style="margin-top:2mm">Then restart the API so the mirror is rebuilt.</p>
  </div>

  <div class="footer-note">
    Run-sheet, troubleshooting table and project-level questions are in
    <span class="mono">docs/DEMO.md</span>. This document covers the interface;
    that one covers the recording.
  </div>
</section>

</body></html>"""


def main() -> int:
    print("reading artifacts and screenshots...")
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

    print(f"wrote {OUTPUT_PDF.relative_to(PROJECT_ROOT)}  "
          f"({OUTPUT_PDF.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
