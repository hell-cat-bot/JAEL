# What the numbers actually mean

Written from scratch, no background assumed. Every example uses our real numbers
from `reports/v1_SMOKE.json` so you can tie it back to the tables.

---

## The setup, as a queue

The system looked at **3,474 loan applications**. Of those, **131 turned out to be
fraud**. So:

**Base rate = 131 / 3,474 = 3.77%.**

That means if you closed your eyes and flagged applications at random, about 1 in
27 of your flags would be a real fraud. Everything below measures how much better
than that we did.

Now imagine the system scores every application from 0 to 1 and sorts them into a
queue, most suspicious first. The case team can realistically review the **top 5%**
— about 174 applications. Every metric below is a different question about that
queue.

---

## Recall@5% — the one that matters operationally

> Of all 131 frauds, how many are sitting inside the top 5% of the queue?

Our best model: **0.702**, meaning **92 of the 131 frauds are caught** while the
team reviews 174 applications. The other 39 slip through.

This is the number to lead with in a demo, because it translates directly into
work: *review 174 files, catch 92 frauds.* A judge understands that instantly.

For comparison, guessing at random would put 5% of the frauds in that window, so
about 7.

---

## Lift — how much better than guessing

**Lift = (hit rate in your flagged set) ÷ (base rate).**

Base rate is 3.77%. Our model's **lift of 19.0×** means the applications it flags
are 19 times more likely to be fraudulent than a randomly picked application.

A lift of **1.0× means you have achieved literally nothing** — you are exactly as
good as a coin flip. This matters because of the bug we found: a misconfigured
model scored lift 1.0×, and we briefly mistook that for "there is no signal in
these features." The model wasn't finding nothing; it was broken.

---

## AUC-PR — the headline score

Also called *average precision*. One number summarising the whole queue, from
"review 10 files" all the way to "review everything."

Intuition: walk down the queue one application at a time. Every time you hit a
real fraud, ask *"of everything I've seen so far, what fraction was actually
fraud?"* Average those fractions. That's AUC-PR.

- **AUC-PR = base rate (0.0377)** → the queue is random ordering. Useless.
- **AUC-PR = 1.0** → every fraud sits above every legitimate application. Perfect.
- **Our 0.717** → strongly ordered, far from random.

Why we use this and not accuracy: with 3.77% fraud, a model that flags *nothing*
is 96.2% accurate and completely worthless. Accuracy is meaningless when one class
is rare. AUC-PR cannot be gamed that way.

---

## AUC-ROC — and why we mostly ignore it

A more familiar metric, but **flattering at low base rates**. Our model scores
**0.976** on AUC-ROC and **0.717** on AUC-PR. The same model, the same predictions
— AUC-ROC just counts things differently and is generous when positives are rare.

We report it because published papers do, so you can compare. We do not lead with
it, because 0.976 sounds like a solved problem and 0.717 is closer to the truth.

**When comparing to research papers, check which one they used.** Many fraud papers
report AUC-ROC only, which makes their numbers look better than ours for reasons
that have nothing to do with model quality.

---

## Confidence intervals — why "0.717" is softer than it looks

We only have **131 frauds**. That is not many. If we reshuffled which applications
were fraudulent (statistically speaking), the score would move around.

A **bootstrap** does exactly that: it repeatedly resamples the data and recomputes
the score, 2,000 times, to see how much it varies.

Our result: **95% CI [0.688, 0.815]**. Read that as *"the true value is probably
somewhere in here."* The spread is about ±0.06, which is wide. So when someone
says 0.717 versus 0.754, that difference is **smaller than the noise** and should
not be treated as a real improvement.

This is why the doubts log insists on quoting intervals, not point estimates.

---

## Ring-disjoint split — the number that protects us from lying to ourselves

To test a model you train it on some data and test it on data it has not seen. The
obvious way is to split applications randomly.

**With fraud rings, random splitting is a trap.** Take a ring of 14 applications.
A random split puts roughly 11 in training and 3 in test. The model has already
seen 11 members of that ring. When asked about the remaining 3, it is not
*detecting a new ring* — it is *recognising an old one*. It scores them perfectly.
Your test score is spectacular and completely fictional, because in production you
will never have seen 11 members of a ring before being asked about the other 3.

**Ring-disjoint splitting** forces every application that shares a device, bank
account, person or guarantor with another into the same group. A ring is then
*entirely* in training or *entirely* in test — never split.

The cost, measured:

| Split | AUC-PR |
|---|---|
| Random (conventional, leaky) | 0.854 |
| Ring-disjoint (honest) | 0.754 |

**That 0.100 gap is the size of the lie.** We report the lower number. Showing
that we measured the gap, and gave up the better-looking figure, is worth more
credibility than the higher number would have been.

---

## Nested CV — not grading your own homework

Models have settings you choose (how deep the trees grow, how small a group they
allow). You normally try several and keep the best.

**If you try them on the same data you then report, you have cheated** — slightly,
and accidentally, but really. You have used the test data to make a choice, so the
test is no longer a fair test. It is like seeing the exam questions before
choosing which textbook chapter to revise.

**Nested cross-validation** fixes it: inside each training portion, the settings
are chosen using only *that* portion. The test portion is never looked at during
the choice.

The cost, measured: **0.754 → 0.717**. That 0.037 is the amount we were
unwittingly flattering ourselves by.

So when you see two numbers for the same model:
- **0.754** = settings picked with the test data in view. Do not quote.
- **0.717** = settings picked honestly. **This is the headline.**

---

## The shuffled-label control — the null test

The most important audit, and the easiest to explain.

Take the fraud labels and **shuffle them randomly between applications**, keeping
the same total count. Now nothing is real: the "frauds" are arbitrary. Retrain the
model on the shuffled labels and test it.

**A model with no leakage must score at chance**, because there is nothing to find.

Our result: AUC-PR **0.041** against a base rate of **0.0377**. That is chance.
**Pass.**

If this test had come back high, it would mean the model was picking up something
other than fraud — a leak, or an artefact of how we built the folds. We actually
hit exactly that once: a bug in the shuffling made the control report 0.242 and
flagged a leak that did not exist. The bug was in the test, not the model.

---

## The best-single-feature floor — the "is this even impressive?" test

Before believing a fancy model, check what **one column on its own** achieves.

Ours: `n_guarantors` — simply counting how many guarantors an application has,
used directly as a score — reaches **AUC-PR 0.233 (6.2× lift)**.

So the honest comparison is:

| | AUC-PR | Lift |
|---|---|---|
| One column, no model at all | 0.233 | 6.2× |
| Full 86-feature model | 0.717 | 19.0× |

That is the real story: **the graph takes you from 6× to 19×.** The earlier claim
of "1× to 17.7×" was wrong, because the 1× came from a broken baseline, not from
the data being uninformative.

This test now runs automatically on every execution, so a broken baseline can
never flatter the model again.

---

## Quick reference

| Term | Plain meaning | Our value | "Good" looks like |
|---|---|---|---|
| Base rate | frauds ÷ total | 3.77% | — |
| Recall@5% | frauds caught in the top 5% of the queue | 0.702 | higher |
| Lift | how many times better than guessing | 19.0× | 1.0 = useless |
| AUC-PR | quality of the whole queue | 0.717 | base rate = useless, 1.0 = perfect |
| AUC-ROC | the flattering cousin | 0.976 | ignore at low base rates |
| 95% CI | where the true value probably sits | [0.688, 0.815] | narrower |
| Random split | conventional, leaks rings | 0.854 | do not report |
| Ring-disjoint | honest | 0.754 | report this |
| Nested CV | settings chosen honestly | 0.717 | **the headline** |
| Shuffled-label control | must land at chance | 0.041 | ≈ 0.0377 |
| Best single feature | the floor | 0.233 | must beat it |

---

## The three sentences to remember

1. **Review the top 5% of the queue and you catch 70% of the fraud** — 92 of 131.
2. **A flagged application is 19× more likely to be fraud than a random one**,
   where 1× would mean the system does nothing.
3. **Every flattering version of these numbers was measured and then given up** —
   random splitting (0.854), self-graded tuning (0.754) — and we report 0.717.
