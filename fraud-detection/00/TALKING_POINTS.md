# Talking Points — Real-Time Fraud Detection System

## 30-second pitch

“I built an end-to-end fraud-detection system on 100,000 synthetic
payment transactions with a 1.85 % fraud rate. It covers the whole
lifecycle: a seeded data generator with leakage-safe rolling features,
a shared feature-engineering module used by both training and the API,
three models compared on PR-AUC, and — the part most portfolio projects
skip — a cost-based threshold analysis: a missed fraud costs the
transaction amount, a false alarm costs a $10 analyst review. At the
cost-optimal threshold the system catches 95.7 % of fraud at 74.8 %
precision and cuts the expected loss by 97.6 % versus doing nothing.
It serves predictions through a FastAPI endpoint and includes a PSI
drift monitor. The logistic regression actually beat the random forest
and the gradient boosting — because the engineered features carry most
of the signal.”

## Numbers to know

| Fact | Number |
|---|---|
| Dataset | 100,000 transactions, 4,000 users, 90 days |
| Fraud rate | 1.85 % (1,845 fraud rows) |
| Split | stratified 80/20 → 80k train / 20k test |
| Best model | Logistic regression (selected by 3-fold CV PR-AUC) |
| CV PR-AUC — LR / RF / HGB | 0.967 / 0.951 / 0.949 |
| Test PR-AUC / ROC-AUC (LR) | 0.966 / 0.998 |
| Chosen threshold | 0.80 (cost-minimising) |
| At 0.80 — precision / recall / F1 | 0.748 / 0.957 / 0.840 |
| Confusion matrix @ 0.80 | TP 353, FP 119, TN 19,512, FN 16 |
| Baseline cost (no model) | $135,549 on the test set |
| Cost with model @ 0.80 | $3,246 → 97.6 % cheaper |
| Batch scoring (5k rows) | 106 alerts (2.1 %), 87.5 % recall, 66 % precision |
| Drift demo | amount PSI 0.18 (moderate), velocity PSI 0.27 (significant) |
| API example | $940, 3 AM, online, foreign → probability 1.0, “flag” |

## Interviewer Q&A

**1. Why is accuracy the wrong metric here?**
With 98.15 % legitimate traffic, a model that approves everything is
98.15 % accurate and catches zero fraud. I optimised PR-AUC for model
selection and a priced confusion matrix for the operating point.

**2. Why PR-AUC over ROC-AUC?**
ROC-AUC is dominated by the enormous true-negative mass, so it stays
near 0.998 for all three models and barely discriminates between them.
Precision-recall focuses on the rare positive class — the only class
the business cares about — and separated the models clearly
(0.949–0.967 in CV).

**3. How did you handle the class imbalance?**
`class_weight='balanced'` in all three models, stratified splits and
CV so every fold keeps the 1.85 % rate, and evaluation on precision /
recall / PR-AUC instead of accuracy. I did *not* use SMOTE — with
engineered numeric features and class weights it wasn't needed, and
synthetic oversampling can distort the very amount distributions the
model relies on.

**4. Why did logistic regression beat the ensembles?**
The fraud signal lives in a few strong, roughly monotonic engineered
features — amount-vs-user-average ratio, velocity, night flag — which
a linear model on standardised features captures well. The forest had
better precision at the default 0.5 threshold (0.91), but model
selection was done on threshold-free CV PR-AUC because the threshold
gets tuned afterwards anyway. Lesson: feature quality beats model
complexity surprisingly often.

**5. How did you choose the 0.80 threshold?**
Priced every candidate threshold on the test predictions: missed
fraud costs its full amount, each false positive costs a $10 review.
The sweep bottoms out at 0.80 ($3,246). Move lower and review costs
explode (1,967 false alerts at 0.05); move higher and missed-fraud
cost takes over. It's an explicit, re-runnable business decision, not
a default.

**6. What leakage risks exist in fraud data, and how did you avoid them?**
The big one is rolling features computed with future data — e.g. a
“30-day average” that includes the transaction itself or later ones.
My generator computes velocity and averages causally, only from each
user's earlier transactions. Second risk: target-derived fields in the
raw data — none are passed to `build_features`. Third: random splits
can leak user behaviour across train/test; a stricter version would
use a time-based split, which I list as a next step.

**7. What is train/serve skew and how is it prevented?**
Skew = the features served in production differ subtly from training
(different code path, different encoding). Here both paths import the
*same* `features.py`, and the model bundle stores the exact feature
column order. Tests assert the API scorer's schema equals the training
schema.

**8. How do you know the model still works next month? (drift)**
`drift_check.py` computes PSI on key features against the training
distribution, with the standard 0.10 / 0.25 action bands, plus fraud-
rate tracking. The demo batch (amounts ×1.35, fraud 2.6 %) triggers
“moderate” on amount and “significant” on velocity — exactly the alert
a production monitor should raise before accuracy silently rots.

**9. The API returns probability 1.0 — is the model calibrated?**
On synthetic data the classes are quite separable, so extreme
probabilities appear. In production I'd verify calibration
(reliability curve, Brier score) and add Platt or isotonic calibration,
because the threshold/cost analysis treats the output as a real
probability.

**10. Why is precision only ~0.75 at the chosen threshold?**
Because the cost model says a missed fraud (average loss ≈ $128 per
missed txn in the sweep) is far more expensive than a $10 review.
Spending 119 reviews to catch 353 of 369 frauds is the *cheap* option.
If review capacity were the constraint instead, I'd optimise
precision@capacity — the framework supports either objective.

**11. What would you productionise next?**
Point-in-time-correct feature computation in a streaming feature
store; time-based validation; delayed-label handling (chargebacks
arrive weeks later); champion/challenger rollout with per-segment
thresholds; and an analyst feedback loop so review verdicts become
training labels.

**12. Why synthetic data?**
Real transaction data is PCI-regulated and unobtainable for a
portfolio. I generated data with documented, literature-based fraud
patterns and stated the limitation openly — the engineering
(data → features → threshold economics → serving → monitoring) is the
transferable part, and it all runs end-to-end.

## Design choices, in plain words

- **One features module for training and serving** — no skew, ever.
- **User-relative features** (amount vs. *your* average) — fraud is a
  deviation from the victim's normal, not from the global normal.
- **Cyclical hour encoding** (sin/cos) — midnight and 1 AM are
  neighbours; raw hour numbers pretend they're far apart.
- **Threshold chosen by money, not by convention** — 0.5 is a default,
  not a decision.
- **PSI with action bands** — drift monitoring that ends in a verb
  (investigate / retrain), not just a chart.
