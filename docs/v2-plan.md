# Anomaly detection v2: plan

## v1's question

From v1's task definition and reflection: *can malicious connections in an IoT connection log be identified, using machine
learning without relying on labels, from features engineered out of the raw columns?* v1's answer was "not very accurate":
Isolation Forest (told that 1% of traffic is anomalous) caught 16 of 21,222 malicious connections on capture 34. Preserved as the
git tag `v1-coursework`.

## Correction to my first v2

The first version of v2 moved to *different* captures where the method happened to work, and called that the answer. That
changed the question. v2 now answers v1's question on v1's own data first, and only then asks whether it generalises.

## Hypothesis (stated before the final numbers)

**H1.** Malicious connections in capture 34 can be separated from benign ones without using labels, using features engineered from
the raw columns, and the engineered features (v1 built rolling per-source counts) are what make that possible.

What would count against it: detectors no better than a coin flip; or the answer not depending on the engineered features.

## Part A: v1's question on v1's data (capture 34, 23,145 connections, 92% malicious)

1. **Reproduce v1** and read its result both ways. In a capture that is mostly malicious, the *rare* class is the benign one, so
   what does "anomalous" actually select?
2. **Different methods**, all scored on the same footing:
   - clustering (k-means, 2 to 8 clusters), each cluster labelled afterwards by its majority class;
   - **novelty detection**: learn "normal" from benign connections only (Isolation Forest, one-class SVM), score the rest;
   - a supervised model as a reference ceiling (what the features can support at all).
3. **Honest scoring**: cross-validate on contiguous time blocks (not random rows, because connections come in bursts); report
   AUROC and average precision (accuracy is meaningless at 92/8) and recall at a false-alarm budget.
4. **Test the second half of H1**: drop each feature group (the rolling count, byte and packet counts, port class, protocol and
   state) and see whether the answer changes; list the best single feature; look at what each class actually connects to.
5. **Look at the structure** v1 noticed ("a cluster of anomalies that lies on a line"): project the features and colour by attack.

## Part B: does it travel?

- Train on capture 34 only, score every other capture.
- Hold each mostly-benign capture (20, 21, 42, 44; 0.1% to 11% malicious) out in turn: unsupervised detectors fitted on the others
  versus a supervised baseline. (This is the setting anomaly detection assumes, so it also answers "what if the data fit the method".)

## Non-goals

No deep models, no live traffic, no claim about real networks. The malicious counts in the mostly-benign captures are tiny (6 to
26), so Part B numbers are wide estimates and the README says so.

## Success criteria

The README states H1, the evidence for and against, and a conclusion that follows from the numbers (including the parts of H1 that
do not survive). `python v2/run.py` reproduces every number and figure; the notebook checks itself against `v2/results.json`.
