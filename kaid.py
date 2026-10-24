"""KAID: a knowledge-based AI decision aid for digital-transformation readiness.

The estimation sample is the public Harmonized Latin American Innovation
Surveys database (LAIS; Crespi, Guillard, Salazar, and Vargas, 2022,
https://doi.org/10.60966/hbz4-cm10). The file is downloaded from the
Inter-American Development Bank on first run and is not redistributed.
License of the microdata: CC BY-NC-ND 3.0 IGO.

Readiness is built only from digital markers. Those markers are excluded
from the predictors. The same enterprise never appears in both the training
sample and the holdout.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache" / "LAIS_public.csv"
DATA_URL = "https://data.iadb.org/files/download/1fe5f05a-8b2a-4a4b-badc-9d8cebaf8011"
SEED = 42
MIN_MARKERS = 3
READY_SHARE = 0.5

DIGITAL = ["web", "IS_dep", "dsoftwexp", "dhardwexp", "dictexp"]
DIGITAL_LABELS = {
    "web": "Website",
    "IS_dep": "Informatics and systems department",
    "dsoftwexp": "Software innovation expenditure",
    "dhardwexp": "Hardware innovation expenditure",
    "dictexp": "ICT innovation expenditure",
}

# Each rule fires when any observed item equals 1. All-missing rules stay missing.
RULES = [
    ("creation", "Knowledge creation", ["drdintexp", "RD_dep"]),
    ("acquisition", "External knowledge", ["drdextexp"]),
    ("training", "Training for innovation", ["dtrainexp"]),
    ("organization", "Organizational integration", ["dinnorg"]),
    ("product", "Product innovation", ["dinnpd"]),
    ("process", "Process innovation", ["dinnpc"]),
]
RULE_DEFINITIONS = {
    "creation": "In-house R&D expenditure or a formal R&D department",
    "acquisition": "External R&D expenditure",
    "training": "Innovation-related training expenditure",
    "organization": "Organizational innovation",
    "product": "Product innovation",
    "process": "Process innovation",
}

# Cooperation and internet-as-a-source are excluded. In this sample those
# questions were not asked for a large share of firms, and "not asked"
# lines up with the readiness label. Using that gap would credit the
# questionnaire rather than the firm's knowledge routines.
AI_NUMERIC = [
    "drdintexp",
    "RD_dep",
    "drdextexp",
    "dtrainexp",
    "dinnorg",
    "dinnpd",
    "dinnpc",
    "dinnmkt",
    "group",
    "dforeign_Y3",
    "log_empl",
]
AI_CATEGORICAL = ["country", "year", "sector"]

BOOSTING = dict(
    max_depth=4,
    learning_rate=0.06,
    max_iter=400,
    l2_regularization=1.0,
    min_samples_leaf=40,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    random_state=SEED,
)


def ensure_data() -> Path:
    override = os.environ.get("LAIS_CSV")
    if override:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    if CACHE.exists() and CACHE.stat().st_size > 1_000_000:
        return CACHE
    fallback = Path("/tmp/lais/LAIS_public.csv")
    if fallback.exists() and fallback.stat().st_size > 1_000_000:
        return fallback
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading LAIS microdata to {CACHE}")
    urllib.request.urlretrieve(DATA_URL, CACHE)
    return CACHE


def _rule_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for key, _label, columns in RULES:
        block = frame[columns].apply(pd.to_numeric, errors="coerce")
        fired = block.eq(1).any(axis=1)
        known = block.notna().any(axis=1)
        out[key] = np.where(known, fired.astype(float), np.nan)
    return out


def rule_score(frame: pd.DataFrame) -> np.ndarray:
    rules = _rule_frame(frame)
    return rules.mean(axis=1, skipna=True).to_numpy(dtype=float)


def load_sample(path: Path) -> pd.DataFrame:
    columns = [
        "country",
        "firm_id",
        "year",
        "isic4_1d",
        "empl_Y3",
        "coop",
        *DIGITAL,
        *AI_NUMERIC,
    ]
    # log_empl is constructed; empl_Y3 is the source.
    columns = [c for c in columns if c != "log_empl"]
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    for column in ["coop", *DIGITAL, *[c for c in AI_NUMERIC if c != "log_empl"]]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    observed = frame[DIGITAL].notna().sum(axis=1)
    share = frame[DIGITAL].mean(axis=1, skipna=True)
    keep = observed >= MIN_MARKERS
    sample = frame.loc[keep].copy()
    sample["n_digital"] = observed.loc[keep].to_numpy()
    sample["digital_share"] = share.loc[keep].to_numpy()
    sample["ready"] = (sample["digital_share"] >= READY_SHARE).astype(int)
    sample["log_empl"] = np.log1p(sample["empl_Y3"].where(sample["empl_Y3"] >= 0))
    sample["sector"] = sample["isic4_1d"].astype("string").fillna("Unknown")
    sample["country"] = sample["country"].astype("string").str.replace("_", " ", regex=False)
    sample["year"] = sample["year"].astype("Int64").astype(str)
    sample = sample.dropna(subset=["country", "firm_id", "ready"]).reset_index(drop=True)
    sample["enterprise"] = sample["country"].astype(str) + "|" + sample["firm_id"].astype(str)
    leaked = set(DIGITAL).intersection(AI_NUMERIC)
    if leaked:
        raise RuntimeError(f"Digital markers leaked into predictors: {sorted(leaked)}")
    return sample


class Design:
    """One-hot encode country, year, and sector. Leave missing numerics missing."""

    def __init__(self, exclude: tuple[str, ...] = ()) -> None:
        self.exclude = set(exclude)
        self.columns_: list[str] | None = None

    def _matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        numeric_columns = [column for column in AI_NUMERIC if column not in self.exclude]
        numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
        cats = pd.get_dummies(frame[AI_CATEGORICAL].astype(str), prefix=AI_CATEGORICAL, dtype=float)
        return pd.concat([numeric.reset_index(drop=True), cats.reset_index(drop=True)], axis=1)

    def fit(self, frame: pd.DataFrame) -> "Design":
        self.columns_ = self._matrix(frame).columns.tolist()
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.columns_ is None:
            raise RuntimeError("Design.fit must run before transform")
        matrix = self._matrix(frame).reindex(columns=self.columns_, fill_value=0.0)
        return matrix.to_numpy(dtype=float)


class KnowledgeLayer:
    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "KnowledgeLayer":
        scores = rule_score(frame)
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.calibrator.fit(scores, target)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.clip(self.calibrator.predict(rule_score(frame)), 0.0, 1.0)


class AILayer:
    def __init__(self, exclude: tuple[str, ...] = ()) -> None:
        self.exclude = exclude

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "AILayer":
        self.design = Design(self.exclude).fit(frame)
        self.model = HistGradientBoostingClassifier(**BOOSTING)
        self.model.fit(self.design.transform(frame), target)
        self.feature_names_ = self.design.columns_
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        probability = self.model.predict_proba(self.design.transform(frame))[:, 1]
        return np.clip(probability, 0.0, 1.0)


def metrics(target: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    predicted = (probability >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(target, probability)),
        "f1": float(f1_score(target, predicted, zero_division=0)),
        "precision": float(precision_score(target, predicted, zero_division=0)),
        "recall": float(recall_score(target, predicted, zero_division=0)),
        "balanced_accuracy": float(
            0.5
            * (
                recall_score(target, predicted, pos_label=1, zero_division=0)
                + recall_score(target, predicted, pos_label=0, zero_division=0)
            )
        ),
        "brier": float(brier_score_loss(target, probability)),
    }


def best_threshold(target: np.ndarray, probability: np.ndarray) -> float:
    grid = np.linspace(0.05, 0.95, 37)
    scores = [f1_score(target, probability >= cut, zero_division=0) for cut in grid]
    return float(grid[int(np.argmax(scores))])


def best_weight(target: np.ndarray, knowledge: np.ndarray, ai: np.ndarray) -> float:
    grid = np.linspace(0.0, 1.0, 21)
    best_w, best_brier = 1.0, np.inf
    for weight in grid:
        fused = weight * ai + (1.0 - weight) * knowledge
        score = brier_score_loss(target, fused)
        if score < best_brier - 1e-12:
            best_brier = float(score)
            best_w = float(weight)
    return best_w


def out_of_fold(frame: pd.DataFrame, target: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    knowledge = np.full(len(frame), np.nan)
    ai = np.full(len(frame), np.nan)
    splitter = GroupKFold(n_splits=5)
    for train_idx, valid_idx in splitter.split(frame, target, groups):
        knowledge_model = KnowledgeLayer().fit(frame.iloc[train_idx], target[train_idx])
        ai_model = AILayer().fit(frame.iloc[train_idx], target[train_idx])
        knowledge[valid_idx] = knowledge_model.predict(frame.iloc[valid_idx])
        ai[valid_idx] = ai_model.predict(frame.iloc[valid_idx])
    if not np.isfinite(knowledge).all() or not np.isfinite(ai).all():
        raise RuntimeError("Out-of-fold probabilities were not finite")
    return knowledge, ai


def grouped_importance(model: AILayer, frame: pd.DataFrame, target: np.ndarray) -> list[dict]:
    matrix = model.design.transform(frame)
    baseline = roc_auc_score(target, model.model.predict_proba(matrix)[:, 1])
    names = model.feature_names_
    drops = []
    rng = np.random.default_rng(SEED)
    families = {name: [i for i, column in enumerate(names) if column == name or column.startswith(f"{name}_")] for name in AI_NUMERIC + AI_CATEGORICAL}
    for family, indexes in families.items():
        if not indexes:
            continue
        scores = []
        for _ in range(8):
            shuffled = matrix.copy()
            order = rng.permutation(len(shuffled))
            shuffled[:, indexes] = shuffled[order][:, indexes]
            scores.append(baseline - roc_auc_score(target, model.model.predict_proba(shuffled)[:, 1]))
        drops.append({"feature": family, "auc_drop": float(np.mean(scores))})
    drops.sort(key=lambda item: item["auc_drop"], reverse=True)
    return drops


def decision_band(probability: float, prevalence: float) -> str:
    if probability >= 0.50:
        return "Prioritize"
    if probability >= prevalence:
        return "Develop"
    return "Hold"


def plot_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 6.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 68)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def box(x, y, w, h, title, body, face):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.4,rounding_size=1.2",
            linewidth=1.1,
            edgecolor="#1F2933",
            facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=11, fontweight="bold", color="#102A43")
        ax.text(x + w / 2, y + h * 0.34, body, ha="center", va="center", fontsize=8.5, color="#243B53", linespacing=1.25)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#334E68"))

    box(28, 54, 44, 10, "Enterprise innovation data", "R&D, training, and innovation outputs\nDigital markers are held out of this layer", "#F0F4F8")
    box(4, 30, 40, 16, "Knowledge layer", "Six routines from the knowledge-based view\nS = share of observed routines that fire\nCalibrated to a probability on training firms", "#D9E8F6")
    box(56, 30, 40, 16, "AI layer", "Gradient boosting on the same innovation items\nMissing survey answers stay missing\nP = probability the firm is ready", "#FCE8D5")
    box(24, 12, 52, 12, "Decision layer", "KAID score = w·P + (1 − w)·S\nw minimizes training Brier score, then the score is frozen", "#E3F4E3")
    box(6, 1.2, 26, 7.2, "Hold", "Score below the training\nreadiness rate", "#FDE8E8")
    box(37, 1.2, 26, 7.2, "Develop", "Between the base rate\nand a 0.50 probability", "#FFF6E0")
    box(68, 1.2, 26, 7.2, "Prioritize", "Probability of readiness\nat least 0.50", "#E5F6E5")
    arrow(50, 54, 24, 46)
    arrow(50, 54, 76, 46)
    arrow(24, 30, 42, 24)
    arrow(76, 30, 58, 24)
    arrow(40, 12, 19, 8.4)
    arrow(50, 12, 50, 8.4)
    arrow(60, 12, 81, 8.4)
    ax.set_title("KAID decision-support architecture", loc="left", fontsize=13, color="#102A43", pad=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc(path: Path, target: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    colors = {"Knowledge layer": "#4C78A8", "AI layer": "#F58518", "KAID": "#1B7F4E"}
    for name, probability in curves.items():
        fpr, tpr, _ = roc_curve(target, probability)
        auc = roc_auc_score(target, probability)
        ax.plot(fpr, tpr, color=colors[name], lw=2.2, label=f"{name}  AUC {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#9AA5B1", lw=1, linestyle="--", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Holdout discrimination")
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def prevalence_table(sample: pd.DataFrame) -> list[dict]:
    rows = []
    for key, _label, _columns in RULES:
        present = sample[key].mean(skipna=True)
        rows.append(
            {
                "rule": key,
                "label": dict((k, lab) for k, lab, _c in RULES)[key],
                "definition": RULE_DEFINITIONS[key],
                "share_firing": float(present),
                "share_observed": float(sample[key].notna().mean()),
            }
        )
    return rows


def main() -> None:
    path = ensure_data()
    sample = load_sample(path)
    rules = _rule_frame(sample)
    sample = pd.concat([sample, rules], axis=1)
    target = sample["ready"].to_numpy(dtype=int)
    groups = sample["enterprise"].to_numpy()
    if set(DIGITAL).intersection(AI_NUMERIC):
        raise RuntimeError("Predictor list overlaps the readiness label")
    if "coop" in AI_NUMERIC or "dinfo_net" in AI_NUMERIC:
        raise RuntimeError("A partly fielded question is in the main predictor list")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_idx, test_idx = next(splitter.split(sample, target, groups))
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    if train_groups & test_groups:
        raise RuntimeError("An enterprise appears in both samples")

    train = sample.iloc[train_idx].reset_index(drop=True)
    test = sample.iloc[test_idx].reset_index(drop=True)
    y_train = train["ready"].to_numpy(dtype=int)
    y_test = test["ready"].to_numpy(dtype=int)

    oof_knowledge, oof_ai = out_of_fold(train, y_train, train["enterprise"].to_numpy())
    weight = best_weight(y_train, oof_knowledge, oof_ai)
    oof_fused = weight * oof_ai + (1.0 - weight) * oof_knowledge
    threshold = best_threshold(y_train, oof_fused)
    knowledge_threshold = best_threshold(y_train, oof_knowledge)
    ai_threshold = best_threshold(y_train, oof_ai)

    knowledge_model = KnowledgeLayer().fit(train, y_train)
    ai_model = AILayer().fit(train, y_train)
    p_knowledge = knowledge_model.predict(test)
    p_ai = ai_model.predict(test)
    p_kaid = weight * p_ai + (1.0 - weight) * p_knowledge
    prevalence = float(y_train.mean())

    figure_dir = ROOT / "figures"
    figure_dir.mkdir(exist_ok=True)
    plot_architecture(figure_dir / "figure1_architecture.png")
    plot_roc(
        figure_dir / "figure2_roc.png",
        y_test,
        {"Knowledge layer": p_knowledge, "AI layer": p_ai, "KAID": p_kaid},
    )

    bands = []
    test_rules = test[[key for key, _lab, _cols in RULES]]
    assigned = [decision_band(float(p), prevalence) for p in p_kaid]
    absent_by_band: dict[str, dict[str, float]] = {}
    masks: dict[str, np.ndarray] = {}
    for name in ("Hold", "Develop", "Prioritize"):
        mask = np.array([band == name for band in assigned])
        masks[name] = mask
        absent: dict[str, float] = {}
        for key, _lab, _cols in RULES:
            known = test_rules.loc[mask, key].dropna() if mask.any() else pd.Series(dtype=float)
            absent[key] = float((known == 0).mean()) if len(known) else 0.0
        absent_by_band[name] = absent
    # External R&D is rare, so "most absent" would name it in every band.
    # The routine reported for a band is the one whose absence most exceeds
    # its absence among Prioritize firms. Prioritize reports its own largest
    # gap among routines that at least 15 percent of firms actually perform.
    common_rules = [row["rule"] for row in prevalence_table(sample) if row["share_firing"] >= 0.15]
    for name in ("Hold", "Develop", "Prioritize"):
        mask = masks[name]
        absent = absent_by_band[name]
        if name == "Prioritize":
            advised = max(common_rules, key=lambda key: absent[key])
        else:
            advised = max(
                common_rules,
                key=lambda key: absent[key] - absent_by_band["Prioritize"][key],
            )
        bands.append(
            {
                "band": name,
                "n": int(mask.sum()),
                "ready_rate": float(y_test[mask].mean()) if mask.any() else None,
                "advised_routine": advised,
                "absent_share": absent,
            }
        )

    holdout = {
        "knowledge": metrics(y_test, p_knowledge, knowledge_threshold),
        "ai": metrics(y_test, p_ai, ai_threshold),
        "kaid": metrics(y_test, p_kaid, threshold),
    }
    importance = grouped_importance(ai_model, test, y_test)
    asked = test["coop"].notna().to_numpy()
    cooperation = None
    if asked.sum() > 80 and len(np.unique(y_test[asked])) == 2:
        train_asked = train["coop"].notna().to_numpy()
        design = Design().fit(train.loc[train_asked])
        extra_train = np.column_stack(
            [design.transform(train.loc[train_asked]), train.loc[train_asked, "coop"].to_numpy(dtype=float)]
        )
        extra_test = np.column_stack(
            [design.transform(test.loc[asked]), test.loc[asked, "coop"].to_numpy(dtype=float)]
        )
        extra_model = HistGradientBoostingClassifier(**BOOSTING)
        extra_model.fit(extra_train, y_train[train_asked])
        extra_probability = extra_model.predict_proba(extra_test)[:, 1]
        without_process = AILayer(exclude=("dinnpc",)).fit(train, y_train)
    process_sensitivity = {
        "auc_without_process_innovation": float(roc_auc_score(y_test, without_process.predict(test))),
    }
    cooperation = {
            "n_train": int(train_asked.sum()),
            "n_test": int(asked.sum()),
            "ready_rate_when_asked": float(y_test[asked].mean()),
            "auc_main_model_on_asked": float(roc_auc_score(y_test[asked], p_ai[asked])),
            "auc_with_cooperation_item": float(roc_auc_score(y_test[asked], extra_probability)),
            "auc_cooperation_item_alone": float(roc_auc_score(y_test[asked], test.loc[asked, "coop"])),
        }
    marker_rows = []
    for column, label in DIGITAL_LABELS.items():
        marker_rows.append(
            {
                "marker": column,
                "label": label,
                "share_positive": float(sample[column].mean(skipna=True)),
                "share_observed": float(sample[column].notna().mean()),
            }
        )

    results = {
        "data": {
            "source": "LAIS public microdata",
            "doi": "10.60966/hbz4-cm10",
            "url": DATA_URL,
            "n_estimation": int(len(sample)),
            "n_enterprises": int(sample["enterprise"].nunique()),
            "n_train_rows": int(len(train)),
            "n_test_rows": int(len(test)),
            "n_train_enterprises": int(len(train_groups)),
            "n_test_enterprises": int(len(test_groups)),
            "ready_rate": float(target.mean()),
            "train_ready_rate": float(y_train.mean()),
            "test_ready_rate": float(y_test.mean()),
            "countries": sorted(sample["country"].dropna().unique().tolist()),
            "years": sorted(sample["year"].unique().tolist()),
            "label": (
                "Ready when at least half of the observed digital markers equal 1, "
                "among firms with at least three observed markers."
            ),
        },
        "markers": marker_rows,
        "rules": prevalence_table(sample),
        "weight_on_ai": weight,
        "thresholds": {
            "knowledge_f1": knowledge_threshold,
            "ai_f1": ai_threshold,
            "kaid_f1": threshold,
            "prioritize_at": 0.50,
            "develop_at_training_prevalence": prevalence,
        },
        "holdout": holdout,
        "bands": bands,
        "importance": importance,
        "process_sensitivity": process_sensitivity,
        "cooperation_sensitivity": cooperation,
        "mean_predicted": {
            "knowledge": float(p_knowledge.mean()),
            "ai": float(p_ai.mean()),
            "kaid": float(p_kaid.mean()),
        },
        "seed": SEED,
    }
    (ROOT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"n": results["data"]["n_estimation"], "weight_on_ai": weight, "holdout": holdout, "bands": bands}, indent=2))


if __name__ == "__main__":
    main()
