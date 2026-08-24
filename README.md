# Network Intrusion Detection with PyTorch

A binary network intrusion classifier trained on NSL-KDD: a logistic regression baseline and a PyTorch MLP, evaluated the way a security team actually cares about — precision, recall, and F1 on both a familiar validation split and a test set containing attack types the model has never seen during training.

## Why this matters

A network intrusion detection system's job is to flag malicious traffic among a much larger stream of normal activity. The metric that matters most here isn't accuracy — it's **recall on the attack class**: a false negative means an actual attack was classified as normal and reached its target undetected, which is a security incident. A false positive just costs an analyst a few minutes of investigation. This project treats recall (and the false-negative rate it implies) as the headline number throughout, not an afterthought next to accuracy.

It also treats generalization to *unseen* attack behavior as a first-class concern, not just held-out accuracy. Signature-based detection can only catch what it already has a signature for; an ML-based approach is only interesting here if it can do meaningfully better against attacks that don't match anything in its training data. NSL-KDD's test set is built specifically to test that.

## Dataset

[NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) — 125,973 labeled training connections, 22,544 test connections, 41 features per connection (protocol, service, byte counts, connection-rate statistics, etc.) plus an attack-type label.

EDA findings that shaped the approach:
- **Binary balance is close to even** (53.5% normal / 46.5% attack) — the imbalance that actually matters shows up one level down: of the 23 individual attack types, `neptune` alone accounts for 41,214 rows while `land`, `buffer_overflow`, and `warezmaster` have 18, 30, and 20 rows respectively. Too few examples for the rarest types to be reliably learnable regardless of model choice — this is why the target here is binary (normal vs. attack), not multi-class.
- **The test set contains 17 attack types that never appear in training** (`worm`, `sqlattack`, `httptunnel`, `mailbomb`, and others) — by design, not by accident. Evaluating against it is a genuine test of generalization to novel attack behavior, not just held-out accuracy on familiar patterns.
- Three categorical features (`protocol_type`, `service`, `flag`) need encoding; `service` alone has 70 distinct values. One constant/dead column (`num_outbound_cmds`) and one dataset-construction artifact (`difficulty_level`) get dropped before training.

## Approach

**Preprocessing** ([src/data.py](src/data.py), [src/preprocess.py](src/preprocess.py)): one-hot encode the 3 categorical features, standardize the ~37 numeric features, stratified 80/20 train/val split. The encoder and scaler are fit on the training split only and applied unchanged to validation and test — fitting on data the model is later evaluated against would leak information into the evaluation and overstate performance.

**Baseline** ([src/baseline.py](src/baseline.py)): `sklearn.linear_model.LogisticRegression` — a single linear decision boundary over the 120 preprocessed features. Exists to answer the question a raw performance number can't: what does a trivial model get, so the MLP's improvement can be measured against something real instead of assumed.

**MLP** ([src/model.py](src/model.py), [src/train.py](src/train.py), [src/evaluate.py](src/evaluate.py)): a deliberately simple feedforward network — 120 inputs → 64 hidden units (ReLU) → 1 output — written in raw PyTorch (`nn.Module`, explicit training loop with `loss.backward()` and manual optimizer steps) rather than a high-level wrapper, trained with `BCEWithLogitsLoss` and Adam for 20 epochs.

**Evaluation** ([src/metrics.py](src/metrics.py)): precision, recall, F1, and a confusion matrix, computed identically for both models so the comparison below is apples-to-apples.

## Results

| | Val precision | Val recall | Val F1 | Test precision | Test recall | Test F1 |
|---|---|---|---|---|---|---|
| **Logistic regression** | 0.976 | 0.963 | 0.969 | 0.917 | 0.625 | 0.743 |
| **MLP** | 0.994 | 0.996 | 0.995 | 0.925 | 0.661 | 0.771 |

**The headline finding isn't which model wins — it's the gap between validation and test for both of them.** Both models generalize almost perfectly to validation data drawn from the same distribution as training (96–99% recall). Both show a large, similarly-sized drop against the test set's novel attack types: logistic regression falls 33.8 points (96.3% → 62.5% recall), the MLP falls 33.5 points (99.6% → 66.1%). The MLP's added nonlinearity gives a real, consistent improvement over the linear baseline (+3.6 points test recall, +2.8 points test F1) — but it does not meaningfully close the generalization gap to attack types neither model has structurally seen before.

That's a more honest and more interesting result than a single leaderboard number: it suggests the gap is coming from the feature representation and the genuine novelty of unseen attack behavior, not from model capacity — and it directly motivates the anomaly-detection framing mentioned below.

## Repo structure

```
src/
  data.py        # load raw NSL-KDD text files into DataFrames
  preprocess.py  # binary target, encode/scale, leak-safe train/val/test split
  metrics.py     # shared precision/recall/F1/confusion-matrix reporting
  baseline.py    # logistic regression baseline
  model.py       # MLP architecture
  train.py       # PyTorch training loop
  evaluate.py    # loads trained weights, reports MLP metrics
data/            # gitignored — see setup below
requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate    # .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

Download NSL-KDD into `data/` (not committed to this repo):
```bash
curl -sL -o data/KDDTrain+.txt "https://raw.githubusercontent.com/Jehuty4949/NSL_KDD/master/KDDTrain%2B.txt"
curl -sL -o data/KDDTest+.txt "https://raw.githubusercontent.com/Jehuty4949/NSL_KDD/master/KDDTest%2B.txt"
```

Run, from `src/`:
```bash
python baseline.py   # logistic regression baseline
python train.py      # trains the MLP, saves mlp_weights.pt
python evaluate.py   # evaluates the trained MLP
```

## What I'd do next

- **Anomaly-detection framing instead of pure supervised classification** — the persistent generalization gap to novel attacks suggests a supervised classifier may be structurally limited here. An autoencoder trained only on normal traffic, flagging high reconstruction error as anomalous, wouldn't need to have seen an attack type during training to catch it — worth comparing directly against this MLP's test-set recall.
- **Extend to CICIDS2017** — a newer, more realistic traffic dataset, as a second benchmark beyond NSL-KDD.
- **Wider/deeper MLP as a cheaper first experiment** before jumping to a different architecture family, to see how much of the gap is genuinely representational versus just under-capacity.
- **Deploy as a live classifier** — wrap the trained model behind a simple API that scores connection records in real time, closer to how this would actually run in a SOC pipeline.
