# Interview Talking Points — Customer Churn Analytics Pipeline

*Read this until you can say it without looking. Every number below is from
the actual run of the code in this repo — if an interviewer asks "what did
you get?", you can answer exactly.*

---

## The 30-Second Pitch

> "I built an end-to-end churn analytics pipeline on a synthetic telco
> dataset of 8,000 customers — I generated the data myself with a seeded
> script because using real customer data in a portfolio isn't appropriate.
> I loaded it into SQLite and answered the business questions in SQL first:
> who churns, and what's different about them. Then I did EDA in Python and
> trained two models — logistic regression and random forest — to predict
> churn. The headline findings: month-to-month customers churn at 37.6%
> versus 18.3% on two-year contracts, first-year customers churn at almost
> 48%, and the logistic regression caught about 72% of churners with a
> test ROC-AUC of 0.75. I finished with concrete retention recommendations
> tied to each finding."

---

## The Numbers to Know Cold

| Fact | Value |
|---|---|
| Dataset size | 8,000 customers, 12 columns |
| Overall churn rate | 30.2% (2,417 churned) |
| Churn — month-to-month | 37.6% |
| Churn — one-year contract | 22.8% |
| Churn — two-year contract | 18.3% |
| Churn — first 12 months | 47.8% |
| Churn — 49+ months tenure | 9.5% |
| Churn — 3+ support calls | 39.0% (vs 23.1% for 0 calls) |
| Churn — 2+ late payments | 38.3% (vs 24.5% for none) |
| Avg monthly charges | churned $62.76 vs retained $58.09 |
| Avg tenure | churned 22.0 mo vs retained 33.7 mo |
| Revenue represented by churners | $151,693/month (~$1.82M/year) |
| Logistic regression | ROC-AUC 0.7467, Recall 0.7184, F1 0.5712, Precision 0.4740, Accuracy 0.6744 |
| Random forest | ROC-AUC 0.7073, Recall 0.5528, F1 0.5115, Accuracy 0.6813 |
| Top features (RF) | total charges, tenure, monthly charges, support calls, late payments |

---

## Likely Questions & Strong Answers

**Q: Why synthetic data? Isn't that fake?**
A: "Yes, deliberately. Real customer data is private — I can't put a real
company's customers on my public GitHub. So I generated a dataset with the
same structure and the same *kinds* of patterns real telco data shows:
contract and tenure effects, support-call and late-payment signals. The
README says clearly that it's synthetic. The skills being demonstrated —
SQL, EDA, modelling, evaluation — are identical to what I'd do with real
data on the job, and everything is seeded so anyone can reproduce my
numbers exactly."

**Q: Walk me through your pipeline.**
A: "Four scripts. `generate_data.py` creates the CSV. `etl_sql.py` loads it
into SQLite as a raw table, then builds a cleaned analytics table in SQL —
trimming text fields, adding a tenure band and a high-value flag — and runs
five business queries: churn by contract, by tenure band, charges for
churned vs retained, support-call impact, late-payment impact. `eda.py`
makes four charts. `model.py` trains and evaluates the models and saves the
metrics and a feature-importance chart."

**Q: Why SQL *and* Python? Why not just do everything in pandas?**
A: "Because that's how the job actually works. The business questions —
churn rate by segment — are set-based questions; SQL answers them in five
lines and runs where the data lives. Python is better for visualisation and
modelling. Also, writing the cleaning logic as a SQL table means the clean
data is reusable by anyone, not trapped inside one notebook."

**Q: Why these features?**
A: "I grouped them by what they represent. Contract and payment method are
*commitment* signals. Tenure and total charges are *relationship history*.
Support calls and late payments are *experience/friction* signals — the
customer is struggling or unhappy. Monthly charges is *price exposure*. I
wanted features a business can actually act on: you can't change someone's
age, but you can fix their support experience or move them to a better
contract."

**Q: Why two models? Why did logistic regression win?**
A: "Logistic regression is my interpretable baseline — coefficients tell you
direction and size of each effect. Random forest is the non-linear check: if
there are interactions or thresholds, it should find them. Here the linear
model won on ROC-AUC, 0.75 vs 0.71, and especially on recall. My read: the
underlying relationships are mostly monotonic — more support calls, more
churn; longer tenure, less churn — so a linear decision boundary generalises
better on 8,000 rows, and the forest's extra flexibility mostly fits noise.
The lesson I took: always benchmark the simple model properly before
reaching for a complex one."

**Q: How did you handle class imbalance?**
A: "The data is about 70/30 retained-to-churned. Three things. First, I used
a stratified split so train and test keep the same ratio. Second, both
models use `class_weight='balanced'`, which up-weights the minority
(churner) class during training so the model doesn't just learn 'predict
retained'. Third — and most important — I didn't judge the models by
accuracy. A model that predicts 'no churn' for everyone scores ~70%
accuracy and is useless. I focused on recall and ROC-AUC, because for
retention, missing a churner is more expensive than a false alarm — a false
alarm costs a discount offer; a missed churner costs the whole customer."

**Q: Precision is only 0.47. Isn't that bad?**
A: "It's the trade-off I chose deliberately. At precision 0.47, about half
the customers we flag actually churn — but we catch 72% of all churners. If
a retention offer costs, say, $20 and a saved customer is worth $60/month,
that trade is profitable. If offers were expensive, I'd raise the decision
threshold to trade recall for precision. The right threshold is a business
cost decision, not a modelling default — that's exactly the conversation I'd
have with the retention team."

**Q: Your top feature is total_charges. Any problem with that?**
A: "Good catch — yes, it needs care. Total charges is roughly tenure times
monthly charges, so it's partly a proxy for tenure, and the two are
correlated (you can see it in the heatmap). It's not leakage — it's known at
prediction time — but I wouldn't tell a stakeholder 'total charges causes
churn'. In a real project I'd test dropping it to see how much is genuinely
tenure. Feature importance shows what the model uses, not what causes the
outcome."

**Q: How did you validate the model?**
A: "Two layers. Five-fold stratified cross-validation on the training set —
logistic regression averaged 0.7595 ROC-AUC across folds, with a tight
standard deviation, so it's stable — and then a final evaluation on a
held-out 20% test set the model never saw during training or CV. I report
the test numbers as the headline results."

**Q: What would you do next / with real data?**
A: "Four things. One: time-based validation — train on earlier months, test
on later — because random splits can flatter you when behaviour drifts.
Two: threshold tuning against an explicit cost model. Three: SHAP values so
every flagged customer comes with human-readable reasons the retention team
can use on a call. Four: productionise it as a monthly scoring job with the
reason codes attached."

**Q: What was the hardest part / what went wrong?**
A: "Honestly, the modelling was the easy part. The discipline was in the
setup: making the generator produce *plausible* patterns without hard-coding
the exact answers, keeping every step seeded and reproducible, and resisting
judging by accuracy. Also the finding that the simple model won — my first
instinct was that the random forest 'should' be better, and the data
disagreed. Trusting the evaluation over my instinct is the habit this
project reinforced."

---

## Design Choices, Explained Simply

- **SQLite**: zero-setup, file-based, and it's real SQL — the same queries
  port to Postgres/BigQuery with minor changes. Right-sized for 8,000 rows;
  I'd use a warehouse engine for millions of rows.
- **Seed = 42 everywhere**: anyone who clones the repo gets byte-identical
  data and identical metrics. Reproducibility is a professional habit.
- **Stratified split**: guarantees the test set has the same 30.2% churn rate
  as the whole data — otherwise metrics wobble for a silly reason.
- **One-hot encoding + scaling in a Pipeline**: preprocessing lives *inside*
  the model pipeline, so there's no way to accidentally leak test-set
  statistics into training (a classic beginner mistake), and the same
  pipeline scores new customers in production shape.
- **`class_weight='balanced'`** instead of SMOTE/oversampling: simpler, no
  synthetic rows, works for both models, easy to explain.
- **Metrics saved to a file, charts saved to disk**: outputs are artefacts a
  stakeholder can open, not numbers trapped in a notebook scrollback.

## If You Get Stuck on a Question

It's fine to say: *"In this project I did X; with real company data I'd also
check Y."* Interviewers reward knowing the boundary of your project more
than bluffing past it. Never claim the data is real, and never quote a
metric you can't trace to `outputs/model_metrics.txt`.
