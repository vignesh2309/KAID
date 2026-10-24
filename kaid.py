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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
        "dinfo_net",
        *DIGITAL,
        *AI_NUMERIC,
    ]
    # log_empl is constructed; empl_Y3 is the source.
    columns = [c for c in columns if c != "log_empl"]
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    for column in ["coop", "dinfo_net", *DIGITAL, *[c for c in AI_NUMERIC if c != "log_empl"]]:
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


class LogisticLayer:
    """Linear baseline. Median imputation is used only here; boosting keeps missingness."""

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> "LogisticLayer":
        self.design = Design().fit(frame)
        self.model = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)),
            ]
        )
        self.model.fit(self.design.transform(frame), target)
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


def out_of_fold(frame: pd.DataFrame, target: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    knowledge = np.full(len(frame), np.nan)
    ai = np.full(len(frame), np.nan)
    logistic = np.full(len(frame), np.nan)
    splitter = GroupKFold(n_splits=5)
    for train_idx, valid_idx in splitter.split(frame, target, groups):
        train_frame = frame.iloc[train_idx]
        valid_frame = frame.iloc[valid_idx]
        y_fold = target[train_idx]
        knowledge[valid_idx] = KnowledgeLayer().fit(train_frame, y_fold).predict(valid_frame)
        ai[valid_idx] = AILayer().fit(train_frame, y_fold).predict(valid_frame)
        logistic[valid_idx] = LogisticLayer().fit(train_frame, y_fold).predict(valid_frame)
    if not np.isfinite(knowledge).all() or not np.isfinite(ai).all() or not np.isfinite(logistic).all():
        raise RuntimeError("Out-of-fold probabilities were not finite")
    return knowledge, ai, logistic


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


RULE_TICKS = {
    "creation": "Knowledge\ncreation",
    "acquisition": "External\nR&D",
    "training": "Training",
    "organization": "Organizational\nintegration",
    "product": "Product\ninnovation",
    "process": "Process\ninnovation",
}
BAND_COLORS = {"Hold": "#9B3A3A", "Develop": "#C47B2B", "Prioritize": "#1F7A4D"}


def _style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#243B53")
    ax.yaxis.label.set_color("#243B53")
    ax.xaxis.label.set_color("#243B53")


def plot_decision(path: Path, frame: pd.DataFrame, assigned: list[str], band_rows: list[dict], base_rate: float) -> None:
    bands = ["Hold", "Develop", "Prioritize"]
    keys = [key for key, _label, _columns in RULES]
    assigned_array = np.array(assigned)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9), gridspec_kw={"width_ratios": [1.45, 1]})
    fig.patch.set_facecolor("white")

    ax = axes[0]
    x = np.arange(len(keys))
    width = 0.24
    for offset, band in enumerate(bands):
        mask = assigned_array == band
        heights = []
        for key in keys:
            known = frame.loc[mask, key].dropna()
            heights.append(float((known == 1).mean()) if len(known) else 0.0)
        ax.bar(
            x + (offset - 1) * width,
            heights,
            width,
            label=band,
            color=BAND_COLORS[band],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([RULE_TICKS[key] for key in keys], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of firm-years in which the routine fires")
    ax.set_title("(a)  Knowledge routines inside each decision band", loc="left", fontsize=11, color="#102A43", pad=18)
    ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.01), borderaxespad=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"{int(value * 100)}%"))
    ax.grid(axis="y", color="#E4E7EB", zorder=0)
    _style(ax)

    ax = axes[1]
    lookup = {row["band"]: row for row in band_rows}
    rates = np.array([lookup[band]["ready_rate"] for band in bands])
    low = np.array([lookup[band]["ci_low"] for band in bands])
    high = np.array([lookup[band]["ci_high"] for band in bands])
    xs = np.arange(len(bands))
    ax.bar(xs, rates, color=[BAND_COLORS[band] for band in bands], width=0.62, edgecolor="white", zorder=3)
    ax.errorbar(
        xs,
        rates,
        yerr=np.vstack([rates - low, high - rates]),
        fmt="none",
        ecolor="#102A43",
        elinewidth=1.1,
        capsize=4,
        zorder=4,
    )
    ax.axhline(base_rate, color="#52606D", linestyle="--", linewidth=1, label=f"Holdout base rate {base_rate:.1%}")
    for index, band in enumerate(bands):
        ax.text(index, min(high[index] + 0.045, 0.97), f"n = {lookup[band]['n']:,}", ha="center", va="bottom", fontsize=8, color="#334E68")
    ax.set_xticks(xs)
    ax.set_xticklabels(bands)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share of firm-years coded ready")
    ax.set_title("(b)  Realized readiness, with a 95% interval", loc="left", fontsize=11, color="#102A43", pad=18)
    ax.legend(frameon=False, loc="upper left")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"{int(value * 100)}%"))
    ax.grid(axis="y", color="#E4E7EB", zorder=0)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_roc(path: Path, target: np.ndarray, curves: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    fig.patch.set_facecolor("white")
    style = {
        "Logistic regression": dict(color="#7B8794", linestyle="-", lw=1.8),
        "Knowledge layer": dict(color="#3D5A80", linestyle="-", lw=2.0),
        "AI layer": dict(color="#E07A3D", linestyle="-", lw=2.4),
        "KAID": dict(
            color="#1F7A4D",
            linestyle="--",
            lw=1.6,
            marker="o",
            markevery=0.07,
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=1.1,
            zorder=5,
        ),
    }
    for name, probability in curves.items():
        fpr, tpr, _thresholds = roc_curve(target, probability)
        auc = roc_auc_score(target, probability)
        ax.plot(fpr, tpr, label=f"{name}   AUC {auc:.3f}", **style[name])
    ax.plot([0, 1], [0, 1], color="#9AA5B1", lw=1, linestyle=":", label="Chance   AUC 0.500")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax.grid(color="#E4E7EB")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _enterprise_draws(groups: np.ndarray, n_boot: int, rng: np.random.Generator) -> list[np.ndarray]:
    enterprises = pd.unique(groups)
    rows_of = {enterprise: np.flatnonzero(groups == enterprise) for enterprise in enterprises}
    draws = []
    for _ in range(n_boot):
        chosen = rng.choice(enterprises, size=len(enterprises), replace=True)
        draws.append(np.concatenate([rows_of[enterprise] for enterprise in chosen]))
    return draws


def bootstrap_auc(target: np.ndarray, curves: dict[str, np.ndarray], groups: np.ndarray, n_boot: int = 2000) -> dict:
    rng = np.random.default_rng(SEED)
    draws = _enterprise_draws(groups, n_boot, rng)
    names = list(curves)
    pairs = [("ai", "knowledge"), ("ai", "logistic"), ("knowledge", "logistic"), ("kaid", "ai"), ("kaid", "knowledge")]
    kept_auc = {name: [] for name in names}
    kept_diff = {pair: [] for pair in pairs}
    for rows in draws:
        y = target[rows]
        if np.unique(y).size < 2:
            continue
        aucs = {name: roc_auc_score(y, curves[name][rows]) for name in names}
        for name, value in aucs.items():
            kept_auc[name].append(value)
        for pair in pairs:
            kept_diff[pair].append(aucs[pair[0]] - aucs[pair[1]])

    def pack(samples: list[float], point: float, against_zero: bool) -> dict:
        array = np.asarray(samples, dtype=float)
        low, high = np.quantile(array, [0.025, 0.975])
        record = {
            "point": float(point),
            "ci_low": float(low),
            "ci_high": float(high),
            "n_boot": int(array.size),
        }
        if against_zero:
            tail = min(float(np.mean(array <= 0)), float(np.mean(array >= 0)))
            record["p_two_sided"] = float(min(1.0, 2 * tail))
        return record

    intervals = {}
    for name, samples in kept_auc.items():
        intervals[name] = pack(samples, float(roc_auc_score(target, curves[name])), against_zero=False)
    comparisons = []
    for left, right in pairs:
        point = float(roc_auc_score(target, curves[left]) - roc_auc_score(target, curves[right]))
        record = pack(kept_diff[(left, right)], point, against_zero=True)
        record["contrast"] = f"{left} minus {right}"
        comparisons.append(record)
    return {"auc": intervals, "comparisons": comparisons}


def bootstrap_band_rates(target: np.ndarray, assigned: list[str], groups: np.ndarray, n_boot: int = 2000) -> dict[str, dict]:
    rng = np.random.default_rng(SEED + 1)
    assigned_array = np.array(assigned)
    draws = _enterprise_draws(groups, n_boot, rng)
    out = {}
    for band in ("Hold", "Develop", "Prioritize"):
        samples = []
        for rows in draws:
            chosen = assigned_array[rows] == band
            if not np.any(chosen):
                continue
            samples.append(float(target[rows][chosen].mean()))
        array = np.asarray(samples, dtype=float)
        mask = assigned_array == band
        low, high = np.quantile(array, [0.025, 0.975])
        out[band] = {
            "ready_rate": float(target[mask].mean()),
            "ci_low": float(low),
            "ci_high": float(high),
            "n": int(mask.sum()),
            "n_boot": int(array.size),
        }
    return out


def fielding_report(sample: pd.DataFrame) -> list[dict]:
    """Share of firm-years in which a candidate item was never asked."""
    rows = []
    for column, label in (("coop", "cooperation"), ("dinfo_net", "internet_as_source")):
        missing = sample[column].isna()
        rows.append(
            {
                "item": label,
                "share_not_asked": float(missing.mean()),
                "n_not_asked": int(missing.sum()),
                "ready_rate_when_not_asked": float(sample.loc[missing, "ready"].mean()) if missing.any() else None,
            }
        )
    return rows


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

    oof_knowledge, oof_ai, oof_logistic = out_of_fold(train, y_train, train["enterprise"].to_numpy())
    weight = best_weight(y_train, oof_knowledge, oof_ai)
    oof_fused = weight * oof_ai + (1.0 - weight) * oof_knowledge
    threshold = best_threshold(y_train, oof_fused)
    knowledge_threshold = best_threshold(y_train, oof_knowledge)
    ai_threshold = best_threshold(y_train, oof_ai)
    logistic_threshold = best_threshold(y_train, oof_logistic)

    knowledge_model = KnowledgeLayer().fit(train, y_train)
    ai_model = AILayer().fit(train, y_train)
    logistic_model = LogisticLayer().fit(train, y_train)
    p_knowledge = knowledge_model.predict(test)
    p_ai = ai_model.predict(test)
    p_logistic = logistic_model.predict(test)
    p_kaid = weight * p_ai + (1.0 - weight) * p_knowledge
    prevalence = float(y_train.mean())

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
        "logistic": metrics(y_test, p_logistic, logistic_threshold),
        "knowledge": metrics(y_test, p_knowledge, knowledge_threshold),
        "ai": metrics(y_test, p_ai, ai_threshold),
        "kaid": metrics(y_test, p_kaid, threshold),
    }
    curves = {"logistic": p_logistic, "knowledge": p_knowledge, "ai": p_ai, "kaid": p_kaid}
    inference = bootstrap_auc(y_test, curves, test["enterprise"].to_numpy())
    band_intervals = bootstrap_band_rates(y_test, assigned, test["enterprise"].to_numpy())
    for row in bands:
        row.update({key: band_intervals[row["band"]][key] for key in ("ci_low", "ci_high", "n_boot")})

    figure_dir = ROOT / "figures"
    figure_dir.mkdir(exist_ok=True)
    plot_decision(figure_dir / "figure1_decision.png", test, assigned, bands, float(y_test.mean()))
    plot_roc(
        figure_dir / "figure2_roc.png",
        y_test,
        {
            "Logistic regression": p_logistic,
            "Knowledge layer": p_knowledge,
            "AI layer": p_ai,
            "KAID": p_kaid,
        },
    )
    old_architecture = figure_dir / "figure1_architecture.png"
    if old_architecture.exists():
        old_architecture.unlink()

    importance = grouped_importance(ai_model, test, y_test)
    without_process = AILayer(exclude=("dinnpc",)).fit(train, y_train)
    process_sensitivity = {
        "auc_without_process_innovation": float(roc_auc_score(y_test, without_process.predict(test))),
    }
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
        "fielding": fielding_report(sample),
        "weight_on_ai": weight,
        "thresholds": {
            "logistic_f1": logistic_threshold,
            "knowledge_f1": knowledge_threshold,
            "ai_f1": ai_threshold,
            "kaid_f1": threshold,
            "prioritize_at": 0.50,
            "develop_at_training_prevalence": prevalence,
        },
        "holdout": holdout,
        "inference": inference,
        "bands": bands,
        "importance": importance,
        "process_sensitivity": process_sensitivity,
        "cooperation_sensitivity": cooperation,
        "mean_predicted": {
            "logistic": float(p_logistic.mean()),
            "knowledge": float(p_knowledge.mean()),
            "ai": float(p_ai.mean()),
            "kaid": float(p_kaid.mean()),
        },
        "seed": SEED,
    }
    (ROOT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"n": results["data"]["n_estimation"], "weight_on_ai": weight, "holdout": holdout, "comparisons": inference["comparisons"], "bands": bands}, indent=2))


if __name__ == "__main__":
    main()
