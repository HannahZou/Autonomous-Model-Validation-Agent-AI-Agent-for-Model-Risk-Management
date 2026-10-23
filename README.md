# Autonomous Model Validation Agent

A multi-agent framework that automates end-to-end model validation in line with the Federal Reserve's **SR 11-7** guidance on model risk management. Specialised agents review a model package — data quality, assumptions, backtesting, challenger benchmarking and sensitivity — and draft a structured validation report with severity-rated findings that a human validator reviews before sign-off.

It supports two model families out of the box:

| Model type | Example | What gets validated |
|---|---|---|
| `credit_pd` | Retail unsecured-loan PD scorecard | Development and validation samples, target definition, features, logistic-regression model (`model.json`) |
| `market_var` | Trading-book 1-day 99% historical VaR | Daily price history, portfolio weights, VaR method / confidence / window |

## How it works

```
model package (YAML manifest + data)
        │
        ▼
   Orchestrator ──► Data Quality ──► Assumptions ──► Backtesting ──► Challenger ──► Sensitivity
        │                 │
        │        deterministic checks  →  findings (severity set by thresholds)
        │                 │
        │        LLM narrative layer   →  summary + commentary (cannot change findings)
        ▼
 results.json · summary.md · audit.jsonl · manifest.json (SHA-256 of every input file)
        │
        ▼
 Human review (Streamlit app)  →  accept / edit / reject each finding, with rationale  →  sign-off
        │
        ▼
 Validation report (.docx / .pdf): rating, findings, charts, rejected findings, audit trail
```

**Sample reports** (from the synthetic examples, with an illustrative demo review):
[credit PD](docs/sample_reports/validation_report_credit_pd.pdf) ·
[market VaR](docs/sample_reports/validation_report_market_var.pdf)

**Design principles**

1. **Numbers come from code, not the LLM.** Every statistic and every severity rating is produced by deterministic, unit-tested checks with thresholds in `config.py`. The same data always gives the same findings.
2. **The LLM writes, it doesn't judge.** OpenAI structured outputs (Pydantic schemas) turn findings into report prose. Commentary for finding IDs the model invents is discarded, and severity and evidence cannot be altered.
3. **Only aggregates leave the machine.** The API receives finding statistics and model documentation, never row-level data.
4. **Never blocked by the API.** If the call fails or no key is set, a deterministic template narrative is used and the error is recorded.
5. **A human decides.** Agents only *propose* findings. A validator accepts, edits or rejects each one; rejections and edits need a written rationale, nothing can be signed off while anything is pending, and nothing changes after sign-off.
6. **Everything is auditable.** Each run writes an append-only `audit.jsonl` — agent timings, every review decision with reviewer and rationale, sign-off — so turnaround time is measured, not claimed.

## Week 1: Data Quality agent

| Check | Credit PD | Market VaR | SR 11-7 element |
|---|---|---|---|
| Missing values (by severity band) | ✓ | ✓ | Conceptual soundness |
| Duplicate records / conflicting IDs / duplicate dates | ✓ | ✓ | Conceptual soundness |
| Values outside documented ranges | ✓ | | Conceptual soundness |
| Extreme outliers (robust z-score) | ✓ | | Conceptual soundness |
| Target validity and default counts | ✓ | | Outcomes analysis |
| Target leakage (univariate AUC) | ✓ | | Conceptual soundness |
| Distribution shift (PSI) | ✓ | | Ongoing monitoring |
| Calendar gaps, date order | | ✓ | Conceptual soundness |
| Zero/negative and stale prices | | ✓ | Conceptual soundness |
| Implausible moves (unadjusted splits, bad ticks) | | ✓ | Conceptual soundness |
| History length vs. VaR window + 1-year backtest | | ✓ | Outcomes analysis |

The synthetic example packages contain deliberately injected defects (a post-default "recovery amount" feature, an unadjusted 2-for-1 split, a frozen FX feed, …). Clean versions of the same generators produce no High or Medium findings, which the test suite checks across several random seeds.

## Week 2: Assumptions and Backtesting agents

**Assumptions agent** (SR 11-7: evaluation of conceptual soundness)

| Check | Credit PD | Market VaR |
|---|---|---|
| Independent replication: re-estimate the documented specification and compare PDs | ✓ | |
| Multicollinearity (VIF) | ✓ | |
| Coefficient signs vs documented business expectations | ✓ | |
| Significance of each feature (re-estimated z-statistics) | ✓ | |
| Coefficient stability: re-estimate on the validation sample, flag sign flips | ✓ | |
| Linearity of log-odds (LR test for a squared term, Bonferroni-adjusted) | ✓ | |
| Independence of returns (Ljung-Box) | | ✓ |
| Volatility clustering (Ljung-Box on squared returns) | | ✓ |
| Normality (Jarque-Bera) when the method is parametric | | ✓ |
| Regime representativeness: recent vs window volatility | | ✓ |

**Backtesting agent** (SR 11-7: outcomes analysis)

| Check | Credit PD | Market VaR |
|---|---|---|
| Discrimination: AUC, Gini, KS on development and validation samples | ✓ | |
| Out-of-sample decay in AUC | ✓ | |
| Calibration in the large (observed vs expected defaults, z-test) | ✓ | |
| Calibration across risk deciles (Hosmer-Lemeshow) with decile table | ✓ | |
| Rolling one-day VaR with no look-ahead, exception series | | ✓ |
| Kupiec proportion-of-failures test | | ✓ |
| Christoffersen independence test (exception clustering) | | ✓ |
| Basel traffic light on the last 250 days | | ✓ |

Statistical tests are implemented in `src/mva/stats.py` with numpy/scipy only and unit-tested against known values (e.g. Kupiec LR = 12.96 for 10 exceptions in 250 days at 99%; the traffic-light function reproduces Basel's 0–4 / 5–9 / 10+ zones).

Severity for statistical tests: rejection at 1% is Medium (High where the error is non-conservative, e.g. under-predicted defaults or too many VaR exceptions); rejection only at 5% is Low. A correct model is still rejected at roughly the test's significance level, so the suite measures this: across 24 clean synthetic packages, at most 2 may produce a High finding.

**Model-level issues planted in the examples**

- *Credit*: the developer added `utilization_6m_avg`, a near-copy of `credit_utilization` (VIF ≈ 36), and the validation sample comes from a worse vintage, so the model under-predicts defaults (21.0% observed vs 16.4% predicted). Risk also steps up once debt-to-income exceeds 45%, which the linear model cannot represent (flagged by the linearity test).
- *VaR*: a stress episode makes the last 60 days about three times as volatile; the 250-day historical VaR lags, giving 16 exceptions against 7.4 expected and a yellow traffic light.

## Week 3: Challenger Benchmarking and Sensitivity Analysis agents

**Challenger Benchmarking agent** (SR 11-7: benchmarking against alternative models)

| | Credit PD | Market VaR |
|---|---|---|
| Challengers built by the validator | Gradient boosting, random forest — same features and preprocessing as the champion | EWMA (RiskMetrics, λ = 0.94) and filtered historical simulation |
| Comparison | Validation AUC with a paired-bootstrap 95% interval for the gap, Brier score, Spearman rank agreement | Exceptions, Kupiec, Christoffersen and traffic light on identical days; quantile (tick) loss; current next-day VaR |
| Finding when | A challenger beats the champion's AUC by > 0.01 and the interval excludes zero | A challenger is in a better traffic-light zone (or passes Kupiec where the champion fails) **and** has lower quantile loss; or every challenger's current VaR exceeds the champion's by > 25% |

**Sensitivity Analysis agent** (SR 11-7: sensitivity analysis under conceptual soundness)

| Credit PD | Market VaR |
|---|---|
| One-at-a-time ±1 robust-SD shocks to each model input; flags a single dominant driver | Current VaR with 125 / 250 / 500-day windows and at 97.5% / 99% / 99.5% confidence; expected shortfall at 97.5% |
| Joint adverse scenario: every feature with a documented sign moved in its risk-increasing direction; PDs must rise | Historical component VaR by asset; flags concentration |
| Parameter uncertainty: 500 coefficient draws from the estimated sampling distribution → 90% interval for portfolio PD | Worst historical day and hypothetical scenarios from the manifest (`stress_scenarios`) as multiples of VaR |

**What they find in the examples**

- *Credit*: both tree-based challengers beat the logistic champion on the validation sample (AUC 0.809 vs 0.794, gap +0.015, 95% interval +0.003 to +0.025) because they capture the debt-to-income threshold. Sensitivity analysis also shows the collinearity at work: shocking `utilization_6m_avg` moves PDs twice as much as shocking current utilisation, although they measure the same thing.
- *VaR*: on the same 739 days the champion has 16 exceptions (yellow zone) while EWMA and FHS have 10 each (green) with lower quantile loss. The worst day in the history, a 20% portfolio loss (4.9× current VaR), is the unadjusted stock split the Data Quality agent flagged.

Thresholds were checked against clean synthetic packages: across 4 clean credit and 12 clean market packages neither agent produces a Medium or High finding, which the test suite enforces.

## Week 4: Human review, validation report and turnaround

**Review workflow** (`src/mva/review.py`, used by the app)

- Every finding starts *pending*. The validator can **accept** it (optional note), **edit** it (severity, title, description or recommendation — rationale required) or **reject** it (rationale required), and can reopen a decision until sign-off.
- **Sign-off** requires a reviewer name and zero pending findings (and an explanation if any agent failed). It fixes the overall rating and locks the review.
- **Overall rating** from the reviewed findings: any High → *Unsatisfactory*; any Medium → *Satisfactory with findings*; otherwise *Satisfactory*.
- **Remediation due dates** from the sign-off date: High 90 days, Medium 180, Low 365 (configurable in `ReviewPolicy`).

**Streamlit app** (`app/streamlit_app.py`): run a validation or open an earlier run, filter findings by area / severity / review status, accept / edit / reject in place, sign off, and download the report. A *Run details* tab shows each agent's metrics, the turnaround measures and the audit trail. The app is a thin layer over `review.py`, so it cannot bypass the review rules; the test suite drives it through a stand-in for the Streamlit API.

**Validation report** (`src/mva/report/`): a Word document — cover page, executive summary with overall rating, model and scope (purpose, documented assumptions and limitations, SR 11-7 coverage, SHA-256 of the inputs), findings summary with severity and due dates, detailed findings by area with charts (calibration by decile and challenger AUC for credit; VaR backtest and challenger exceptions for market risk), and appendices for rejected findings, the audit trail and turnaround, and the limitations of the validation. Before sign-off it is marked **DRAFT**. PDF conversion uses LibreOffice when installed.

The executive summary is written by the LLM *after* review, from the reviewed findings only (template fallback; finding IDs it cites are checked). The report labels every AI-written passage as such.

**Turnaround** is read from the audit trail: automated analysis time per agent, active review time (first to last decision), and elapsed time from run start to sign-off. `mva benchmark` times the automated pipeline (median of repeated runs, no API calls):

| Package | Automated run (median of 3) | Slowest agent |
|---|---|---|
| Credit PD (8,160 + 3,000 loans) | 4.8 s | Challenger Benchmarking, 4.5 s (tree models + 300 bootstrap resamples) |
| Market VaR (4 assets, ~1,000 days) | 0.06 s | — |

The analysis itself takes seconds; the human review is what remains, and it is now measured per run. Comparing against a manual validation baseline requires recording that baseline in your own process.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env            # add OPENAI_API_KEY to get LLM narratives

mva make-examples               # writes examples/credit_pd and examples/market_var
mva validate examples/credit_pd --no-llm
mva validate examples/market_var          # uses OpenAI when a key is set
pytest

streamlit run app/streamlit_app.py         # review, sign off, download the report
mva report runs/<run_id> --pdf             # or build the report from the command line
mva turnaround runs/<run_id>
mva benchmark examples/credit_pd examples/market_var --repeat 3
```

Example output:

```
Retail Unsecured Loan PD Scorecard (CR-PD-001)
  Data Quality      8 findings  (0.15s, narrative: template)
    DQ-001  High    Conflicting records share an ID in development data
    DQ-002  High    Values outside documented valid ranges in development data
    DQ-003  High    Suspected target leakage
    DQ-004  High    Feature distribution shift between development and validation samples (significant)
    DQ-005  Medium  Missing values in development data
    DQ-006  Medium  Duplicate records in development data
    DQ-007  Medium  Extreme outliers in development data
    DQ-008  Medium  Default rate differs materially between samples
  Assumptions       3 findings  (0.07s, narrative: template)
    AS-001  High    Multicollinearity among model features (severe)
    AS-002  Medium  Non-linear relationship with log-odds
    AS-003  Low     Features without statistically significant effect
  Backtesting       2 findings  (0.13s, narrative: template)
    BT-001  High    Model under-predicts defaults
    BT-002  Medium  Predicted PDs do not match observed default rates across risk deciles
  Challenger Benchmarking  2 findings  (5.24s, narrative: template)
    CH-001  Medium  Gradient boosting challenger outperforms the champion
    CH-002  Medium  Random forest challenger outperforms the champion
  Sensitivity Analysis  0 findings  (0.07s, narrative: template)
Total: 6 High, 8 Medium, 1 Low  in 5.67s
```

Each run is saved to `runs/<run_id>/` as `results.json` (machine-readable), `summary.md` (quick-look findings), `manifest.json` (what was validated, with file hashes), `audit.jsonl`, `artifacts/` (series behind the charts), and after review `review.json` and the report.

## Model package format

A package is a folder with `model_package.yaml` and the data it references. See `examples/credit_pd/model_package.yaml` and `examples/market_var/model_package.yaml`. Documentation (purpose, intended use, assumptions, limitations) lives in the manifest so agents can test claims against it.

A credit package also points to `model.json`: the delivered logistic regression as data — intercept, coefficients, the preprocessing applied (imputation → caps → transforms) and the expected sign of each coefficient. A JSON specification is safe to load (no pickles), readable in code review, and mirrors how scorecards are documented. A VaR model is fully described by its `var` block (method, confidence, window, horizon); optional `stress_scenarios` define hypothetical one-day price shocks used by the Sensitivity agent.

## Roadmap

- **Week 1** ✅ Project skeleton, model package format, Data Quality agent, LLM narrative layer, orchestrator, audit log
- **Week 2** ✅ Model specification in the package, statistics library, Assumptions agent, Backtesting agent
- **Week 3** ✅ Challenger Benchmarking agent and Sensitivity Analysis agent — all five agents in place
- **Week 4** ✅ Human review workflow and sign-off, Streamlit app, Word/PDF validation report with charts, LLM executive summary, turnaround measurement and benchmark

## Project layout

```
src/mva/
  config.py          thresholds and settings
  schemas.py         Finding, AgentReport, ValidationRun
  package.py         model package manifest and loader
  llm.py             narrative layer (OpenAI structured outputs + template fallback)
  orchestrator.py    runs agents, writes results, manifest snapshot, artifacts and audit trail
  audit.py           append-only audit log
  review.py          human review rules: accept / edit / reject, rating, remediation dates, sign-off
  turnaround.py      timings from the audit trail
  report/            Word report builder, charts, PDF conversion
  stats.py           statistical tests (logit MLE, VIF, Hosmer-Lemeshow, Kupiec, Christoffersen, ...)
  models/            the models under validation (PD scoring, rolling VaR) and challenger VaR models
  agents/            one module per validation agent
  checks/            deterministic tests used by the agents
  synthetic/         synthetic credit and market data generators
app/streamlit_app.py review app
scripts/             regenerate examples and sample reports
docs/sample_reports/ example validation reports
tests/               pytest suite (OpenAI and Streamlit are stood in for)
```

All data in this repository is synthetic.
