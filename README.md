# Network Intrusion Detection with PyTorch

A binary network intrusion classifier trained on NSL-KDD, comparing a logistic regression baseline against a PyTorch MLP. Evaluated on precision, recall, and F1 across a validation split and a test set that includes attack types the model never saw during training.

## Why this matters

An intrusion detection system has to flag malicious traffic inside a much larger stream of normal activity. Accuracy isn't the metric that matters here, recall on the attack class is. A false negative means an attack got classified as normal and reached its target undetected; that's a security incident. A false positive just costs an analyst a few minutes checking something that turned out to be nothing. Recall (and the false-negative rate behind it) is treated as the headline number throughout this project, not accuracy.

Generalization to attack behavior the model hasn't seen before matters just as much. Signature-based detection can only catch what it already has a signature for. An ML-based approach only earns its keep if it does meaningfully better against attacks that don't resemble anything in its training data, and NSL-KDD's test set is built specifically to test that.

## Dataset

[NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html): 125,973 labeled training connections, 22,544 test connections, 41 features per connection (protocol, service, byte counts, connection-rate stats, etc.) plus an attack-type label.

A few things from EDA shaped the approach:

- Binary balance is close to even (53.5% normal / 46.5% attack), but that's not where the real imbalance is. One level down, of the 23 individual attack types, `neptune` alone accounts for 41,214 rows while `land`, `buffer_overflow`, and `warezmaster` have 18, 30, and 20 rows respectively. There just aren't enough examples of the rarest types to learn them reliably, which is part of why the target here is binary (normal vs. attack) rather than multi-class.
- The test set contains 17 attack types that never show up in training (`worm`, `sqlattack`, `httptunnel`, `mailbomb`, among others). That's intentional, not an artifact of the split. Evaluating against it is a real test of generalization to attack behavior the model hasn't seen, not just held-out accuracy on familiar patterns.
- Three categorical features (`protocol_type`, `service`, `flag`) need encoding, and `service` alone has 70 distinct values. One constant column (`num_outbound_cmds`) and one dataset-construction artifact (`difficulty_level`) get dropped before training.

## Approach

**Preprocessing** ([src/data.py](src/data.py), [src/preprocess.py](src/preprocess.py)). One-hot encode the 3 categorical features, standardize the ~37 numeric ones, stratified 80/20 train/val split. The encoder and scaler are fit on the training split only and applied unchanged to validation and test. Fitting on data the model is later evaluated against would leak information into the evaluation and inflate the numbers.

**Baseline** ([src/baseline.py](src/baseline.py)). `sklearn.linear_model.LogisticRegression`, a single linear decision boundary over the 120 preprocessed features. It exists to answer a question a raw performance number can't: what does a trivial model get, so the MLP's improvement is measured against something real instead of assumed.

**MLP** ([src/model.py](src/model.py), [src/train.py](src/train.py), [src/evaluate.py](src/evaluate.py)). A deliberately simple feedforward network: 120 inputs, 64 hidden units with ReLU, 1 output. Written in raw PyTorch (`nn.Module`, an explicit training loop with `loss.backward()` and manual optimizer steps) rather than a high-level wrapper, trained with `BCEWithLogitsLoss` and Adam for 20 epochs.

**Evaluation** ([src/metrics.py](src/metrics.py)). Precision, recall, F1, and a confusion matrix, computed the same way for both models so the comparison below is apples-to-apples.

## Results

| | Val precision | Val recall | Val F1 | Test precision | Test recall | Test F1 |
|---|---|---|---|---|---|---|
| **Logistic regression** | 0.976 | 0.963 | 0.969 | 0.917 | 0.625 | 0.743 |
| **MLP** | 0.994 | 0.996 | 0.995 | 0.925 | 0.661 | 0.771 |

The interesting part isn't which model wins. It's the gap between validation and test for both of them. Both models generalize almost perfectly to validation data drawn from the same distribution as training, 96-99% recall. Both drop hard against the test set's novel attack types: logistic regression falls 33.8 points (96.3% to 62.5% recall), the MLP falls a similar 33.5 points (99.6% to 66.1%). The MLP's nonlinearity does give a real, consistent improvement over the linear baseline (+3.6 points test recall, +2.8 points F1), but it doesn't come close to closing the gap to attack types neither model has seen before.

That result is more useful than a single leaderboard number. It points to the gap coming from the feature representation and the genuine novelty of unseen attacks, not from model capacity, which is what motivates the anomaly-detection idea below.

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
data/            # gitignored, see setup below
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

- **Anomaly-detection framing instead of pure supervised classification.** The generalization gap suggests a supervised classifier might be the wrong tool for catching truly novel attacks. An autoencoder trained only on normal traffic, flagging high reconstruction error as anomalous, wouldn't need to have seen an attack type during training to catch it. Worth comparing directly against this MLP's test recall.
- **Extend to CICIDS2017** as a second, more realistic benchmark beyond NSL-KDD.
- **Try a wider/deeper MLP first**, before jumping to a different architecture family, to see how much of the gap is actually representational versus just under-capacity.
- **Deploy as a live classifier**, wrapped behind a simple API that scores connection records in real time, closer to how this would run in an actual SOC pipeline.
