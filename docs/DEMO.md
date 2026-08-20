# Recording the demonstration

A run-sheet. Follow it in order and the recording takes about five minutes.

---

## Before you press record

**1. Start both servers and wait.**

```powershell
# terminal 1 — API
cd C:\Users\ruthv\OneDrive\Desktop\Projects\syf-fraud-detection\backend
C:\dev\venvs\syf-fraud\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# terminal 2 — console
cd C:\Users\ruthv\OneDrive\Desktop\Projects\syf-fraud-detection\frontend
npm run dev
```

Give the API about fifteen seconds. At startup it loads the model, builds the
SHAP explainer, connects to Postgres and rebuilds the similarity index. Opening
the console before that finishes shows an error card.

**2. Confirm the state is clean.**

```powershell
cd C:\Users\ruthv\OneDrive\Desktop\Projects\syf-fraud-detection
C:\dev\venvs\syf-fraud\Scripts\python.exe scripts\reset_demo.py
```

You want `fraud_vectors 400 (400 seeded, 0 added at runtime)`. If it shows
runtime vectors, a previous rehearsal confirmed a case — clear them, or a
genuine applicant will be escalated on camera for no visible reason:

```powershell
C:\dev\venvs\syf-fraud\Scripts\python.exe scripts\reset_demo.py --vectors-only --apply
```

Then restart the API so the in-process index is rebuilt.

**3. Close everything else.**

This matters more than it sounds. Latency is measured server-side and shown on
screen. With unrelated build watchers running, p50 goes from **46ms to 118ms** —
on a system whose headline claim is real-time decisioning. Check Task Manager
shows CPU near idle before you start.

**4. Warm it up, then reset the view.**

Press Start, let ten or so applications stream through, then Pause and reload
the page. The first few requests are slower while SHAP compiles its kernels;
after that latency settles. The warmed numbers are the honest steady state, not
a trick — you are just not recording the compile.

---

## The recording

### Beat 1 — what you are looking at (30s)

Start on the paused console. Name the three columns left to right: what arrived,
why it was decided, the policy that decided it. Point out the header: model
version, index size, and the green **postgres** indicator.

> "Left is the live queue. Middle is why any given decision was made. Right is
> the policy governing all of them — and everything is being written to Postgres
> as it happens."

### Beat 2 — decisions arriving (45s)

Press **Start**.

Let it run. Call out the metrics strip as it moves — scored, blocked, fraud
caught, and **mean latency**. Let the audience see latency sitting under 100ms.

> "Every one of these is a full decision: the model, an isotonic calibrator, the
> cost policy, and SHAP attributions. Under a tenth of a second."

Point at a row marked **MISSED** if one appears.

> "That one was fraud and we approved it. The ledger shows misses, not just
> catches — a fraud console that only shows its wins is not a console."

Press **Pause**.

### Beat 3 — why a decision was made (60s)

Click a **REVIEW** or **BLOCK** row.

Walk the case detail top to bottom: the probability, the threshold meter showing
where it landed between review and block, then the force bars.

> "These bars are SHAP attributions and they are additive — they sum to the
> model's output. This is not an illustration of the explanation, it *is* the
> explanation. Ordered by magnitude, they become the adverse-action reason codes
> a regulator expects."

Point out the **Analyst briefing** and its `TEMPLATE` / `GEMINI` tag.

> "The language model writes this paragraph from attributions that already
> exist. It cannot change a score, a threshold, or an outcome — and it is off
> the decision path entirely, because it costs three seconds and a decision
> cannot wait for it."

### Beat 4 — the policy is arithmetic (45s)

Move to **Risk appetite** on the right. Drag **cost of blocking a genuine
customer** upward.

> "I have not retrained anything. These thresholds are derived from four
> business costs by minimising expected loss. Raising the cost of a wrongful
> block makes the system measurably less willing to block. Changing the bank's
> risk appetite is arithmetic, not a model release."

Watch the threshold numbers move. Put the slider back.

### Beat 5 — the adaptation loop, and the audit trail (90s)

**This is the strongest beat. Do not rush it.**

Click the **Audit trail** tab on the lower-right panel. Show the counters and
the append-only line.

> "Every decision you just watched is in Postgres. The audit table is
> append-only — enforced by a database trigger, not by convention. Nothing in
> this application can rewrite a recorded decision."

Now select a case in the ledger and press **Confirm fraud**.

Watch two things move: the header `index` count increments, and a new
`analyst.verdict` entry appears in the audit trail.

> "One analyst confirmed one case. That case is now in the vector index — in
> Postgres, not in memory."

Press **Start** again briefly, or re-score the same application via `/docs`.

> "The next application matching that pattern is escalated automatically. No
> retraining, no redeploy. One confirmed case became a detector — and because it
> is in Postgres, it survives a restart. The model's own decision is still shown
> alongside, so an auditor can always see what was decided and what changed it."

### Beat 6 — honesty (30s)

Move to **Model health**.

> "Recall falls from 63% to 48% across the held-out months. What decayed is not
> the model — ROC-AUC barely moves. It is the operating point: fraud prevalence
> rose while the thresholds stayed put. We tested four adaptation strategies and
> none beat doing nothing. That is in the report as a negative result, because a
> report containing only the experiments that worked is not evidence of
> judgement."

---

## If something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| "Cannot reach the API" | API not started, or still loading | Wait 15s; check terminal 1 |
| "API key rejected" | `VITE_API_KEY` ≠ `API_KEY` | Compare `frontend/.env` and `.env` |
| Header shows **no database** | `DATABASE_URL` unset or unreachable | Check `.env`; the console still works, persistence is off |
| Latency reads 300ms+ | Something else is using the CPU | Close it; see step 3 above |
| Briefing says `TEMPLATE` not `GEMINI` | Free-tier quota exhausted | Expected; quota resets daily. The template is the designed fallback — say so rather than hiding it |
| A genuine case escalates unexpectedly | A rehearsal confirmed it as fraud | `reset_demo.py --vectors-only --apply`, restart API |

---

## Questions you should expect

**"Why FastAPI and not Spring Boot?"**
The model is Python — LightGBM and SHAP. Serving it in-process avoids a
cross-process hop on a path budgeted in tens of milliseconds. A Spring Boot
service would have to call this one anyway.

**"Where is the cloud deployment?"**
There isn't one, and that is stated in the report. It runs locally against
managed Postgres. Containerisation and IAM are designed, not built.

**"Is this real data?"**
No — BAF is a CTGAN rendering of a real anonymised bank dataset under
differential privacy. It preserves structure, feature semantics and genuine
temporal drift. The fields are unambiguously credit-product ones: proposed
credit limit, payment plan type, other cards held.

**"Is that not account-opening rather than lending?"**
The dataset's own datasheet says account opening; its fields say credit
origination. In substance it is lending-origination fraud. What it does *not*
contain is post-origination behaviour, so first-party bust-out fraud is out of
scope — and that is stated rather than glossed.

**"Why is accuracy not on the slide?"**
At 1.4% prevalence, a model that predicts "never fraud" scores 98.6%. Accuracy
on this problem reports the base rate.

**"What is the weakest part?"**
The cost model does not price capacity overflow — it charges ₹200 per review
whether or not an analyst exists to do it. That is why a fixed cut-off appears
to beat the alert budget on recall: it is spending hours it does not have.
