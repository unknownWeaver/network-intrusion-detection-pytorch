# Project build prompt — PyTorch network intrusion classifier

Paste this into a new chat (Claude Code works well for this since it can run and iterate on code directly).

---

I want to build a small but real PyTorch project: a network intrusion / traffic classifier trained on the NSL-KDD dataset, sized to finish in a weekend and good enough to put on GitHub and talk through in an interview. This is specifically going on my resume for a ByteDance internship posting focused on ML-based network traffic anomaly detection, so the framing (anomaly/attack detection in network traffic) should stay front and center — not a generic ML exercise.

**Set this up like a real project, not a notebook dump:**
- Initialize a git repo at the start and commit incrementally as we go (EDA, baseline, model, eval, README) — a real commit history is part of what makes this credible, not one giant commit at the end.
- Use a proper structure: `src/` for code (data loading, model, train, evaluate as separate modules, not one script), `notebooks/` only if we need one for EDA, `data/` gitignored (never commit the raw dataset), `requirements.txt` pinned, `.gitignore` set up properly, `README.md` as the front door.
- Use a virtual environment, not a global install.

**My background, so you calibrate correctly:**
- Solid Python, early-stage C++
- I've worked through Karpathy's "Neural Networks: Zero to Hero" — built micrograd, backprop, and an MLP from scratch by hand. I understand autograd, gradients, and training loops conceptually and have implemented them myself.
- I have NOT built a real applied PyTorch project yet — this is the first one.
- Cybersecurity background: Security+, CySA+, SC-900, CMMC RP, AAS in Cybersecurity, currently in a BS in Cybersecurity and Information Assurance. I understand intrusion detection concepts (signatures vs. anomaly-based detection, false positive/negative tradeoffs) but haven't applied ML to them before.

**How I want you to work with me:**
- Don't just hand me a finished script. Walk me through this in stages, explain the reasoning at each decision point (why this architecture, why this loss function, why this metric matters more than accuracy here), and let me write pieces myself where it's reasonable to.
- Check in with me before moving to the next stage instead of dumping the whole solution at once.
- Where something maps back to what I already know from Karpathy's series (autograd, backprop, training loop mechanics), point that out explicitly — I want to connect what I already understand to the applied version.
- Push back if a design choice I suggest is wrong or lazy — don't just validate whatever I propose.

**What I want built, in order:**

1. **Data**: NSL-KDD (not the older/messier CICIDS2017 — save that for a v2 if I want to extend this later). Help me get it, understand the features (protocol type, service, flag, byte counts, etc.), and do a real EDA — class balance, which features actually separate attack traffic from normal traffic, what preprocessing is needed (categorical encoding, normalization).

2. **Baseline**: A simple non-neural baseline first (logistic regression or a decision tree) so I have something to compare the neural net against. A model that "gets 95% accuracy" means nothing without knowing what a trivial baseline gets, especially since NSL-KDD's class balance can make accuracy misleading.

3. **PyTorch model**: A straightforward feedforward MLP — nothing exotic, this isn't the point. Written in raw PyTorch (nn.Module, explicit training loop with autograd), not a high-level wrapper library, so it stays connected to the fundamentals I already built by hand.

4. **Evaluation that actually matters for security**: precision, recall, F1, and a confusion matrix — and a real discussion of why false negatives (missed attacks) are the metric that matters most here, not raw accuracy. I want to be able to explain this tradeoff in an interview, not just report a number.

5. **README** written for a hiring audience: problem statement, why this matters (tie to intrusion detection / SOC use cases), approach, results with the baseline comparison, and an honest "what I'd do next" section (e.g., extending to CICIDS2017, trying a different architecture, deploying it as a live classifier).

**Constraints:**
- Keep the model itself simple — this is a portfolio/interview piece, not a research project. The value is in doing it correctly and being able to explain every part, not in squeezing out marginal accuracy gains.
- Runs locally on my machine, no cloud GPU needed.
- End state: a clean GitHub-ready repo I can link from a resume and application.

Start with step 1 — help me get the dataset and walk me through the EDA.
