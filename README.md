# KAID

A knowledge-based AI decision aid that predicts digital-transformation readiness from enterprise innovation data, plus the short communication that reports it.

The paper is [`paper.md`](paper.md). One script, [`kaid.py`](kaid.py), downloads the public survey file, fits the model, and rewrites the two figures and `results.json`.

## What the model does

KAID scores six knowledge routines (R&D, external R&D, training, organizational innovation, product innovation, process innovation) and fits a gradient-boosting model on the same innovation items. Digital markers are the label only. They are not predictors. The holdout split keeps each enterprise on one side. On the reported split, logistic regression reaches AUC 0.873, the knowledge layer 0.854, and boosting 0.891. Boosting beats both, and blending does not beat boosting (p = 0.223). The practical output is a band:

- **Hold** — do not build the digital offer yet. The missing routine is process innovation.
- **Develop** — build innovation training before another software purchase.
- **Prioritize** — the innovation record can support the offer. The remaining gap is often an R&D routine.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 kaid.py
```

The first run downloads the LAIS public microdata (about 118 MB) from the Inter-American Development Bank into `.cache/`. Set `LAIS_CSV` to use a file you already have. The microdata are not stored in this repository.

## Data

Crespi, G., Guillard, C., Salazar, M., & Vargas, F. (2022). *Harmonized Latin American Innovation Surveys Database (LAIS)*. Inter-American Development Bank. https://doi.org/10.60966/hbz4-cm10

The microdata are licensed CC BY-NC-ND 3.0 IGO. This repository redistributes the code, the figures, and the aggregate metrics, not the firm file.

## Files

| File | Role |
|---|---|
| `kaid.py` | Data, model, figures, metrics |
| `paper.md` | Short communication |
| `results.json` | Holdout numbers written by the script |
| `figures/figure1_decision.png` | Decision bands and knowledge routines |
| `figures/figure2_roc.png` | Holdout ROC curves |
