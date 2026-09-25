"""Figures du README et du rapport, écrites dans figures/.

Couleurs : une par modèle, la même dans toutes les figures (palette de
référence validée, sûre pour les daltoniens sur trois couleurs).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from mplsoccer import VerticalPitch  # noqa: E402

try:
    from .explain import CATEGORICAL, LABELS, NUMERIC, out_of_fold_explanations, sigmoid
    from .train import DATA, REPORTS
except ImportError:  # python src/figures.py
    from explain import CATEGORICAL, LABELS, NUMERIC, out_of_fold_explanations, sigmoid
    from train import DATA, REPORTS

FIGURES = Path(__file__).resolve().parent.parent / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

MODELS = [  # (colonne, libellé, couleur)
    ("xg_logistique", "Logistique", "#2a78d6"),
    ("xg_boosting", "Gradient boosting", "#eb6834"),
    ("xg_statsbomb", "xG StatsBomb", "#1baf7a"),
]
INCREASE, DECREASE = "#2a78d6", "#e34948"  # paire divergente bleu / rouge

CATEGORIES_FR = {
    "Head": "Tête", "Left Foot": "Pied gauche", "Right Foot": "Pied droit", "Other": "Autre",
    "Normal": "Frappe normale", "Volley": "Volée", "Half Volley": "Demi-volée", "Lob": "Lob",
    "Backheel": "Talonnade", "Diving Header": "Tête plongeante", "Overhead Kick": "Retourné",
    "Open Play": "Jeu courant", "Free Kick": "Coup franc direct", "Corner": "Corner direct",
    "Kick Off": "Engagement", "Ground Pass": "Au sol", "Low Pass": "Basse", "High Pass": "Haute",
    "Regular Play": "Attaque placée", "From Counter": "Contre-attaque", "From Corner": "Après corner",
    "From Free Kick": "Après coup franc", "From Throw In": "Après touche",
    "From Goal Kick": "Après 6 mètres", "From Keeper": "Relance du gardien",
    "From Kick Off": "Après engagement",
}


def fr(value) -> str:
    return CATEGORIES_FR.get(value, value)


def number(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")


def describe(feature: str, value) -> str:
    """Valeur lisible d'une feature pour un tir."""
    if feature in CATEGORICAL:
        return fr(value)
    if isinstance(value, (bool, np.bool_)):
        return "oui" if value else "non"
    if feature == "angle":
        return f"{number(np.degrees(value), 0)}°"
    if feature == "defenders_in_triangle":
        return str(int(value))
    return number(value)

BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "grid.color": GRID,
    "grid.linewidth": 1,
    "lines.linewidth": 2,
    "lines.solid_capstyle": "round",
})


def french(decimals: int) -> FuncFormatter:
    """Virgule décimale : 0,25 et non 0.25."""
    return FuncFormatter(lambda value, _: f"{value:.{decimals}f}".replace(".", ","))


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "\u202f")  # espace fine insécable : 5 583


def calibration(predictions: pd.DataFrame, path: Path, n_bins: int = 10) -> None:
    """Un panneau par modèle : xG prédit contre taux de buts observé, 10 tranches.

    Les barres verticales sont des intervalles à 95 % : avec ~136 tirs par
    tranche, un écart à la diagonale plus petit que la barre n'est pas un défaut.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), sharex=True, sharey=True)
    limit = 0.6

    for ax, (column, label, color) in zip(axes, MODELS):
        frame = predictions[["is_goal", column]].dropna()
        bins = pd.qcut(frame[column], n_bins, labels=False, duplicates="drop")
        grouped = frame.groupby(bins).agg(predit=(column, "mean"), observe=("is_goal", "mean"),
                                          n=("is_goal", "size"))
        half_width = 1.96 * np.sqrt(grouped["observe"] * (1 - grouped["observe"]) / grouped["n"])

        ax.plot([0, limit], [0, limit], color=BASELINE, linewidth=1, zorder=1)
        ax.vlines(grouped["predit"], (grouped["observe"] - half_width).clip(lower=0),
                  grouped["observe"] + half_width, color=color, alpha=0.35, linewidth=2, zorder=2)
        ax.plot(grouped["predit"], grouped["observe"], color=color, zorder=3)
        ax.scatter(grouped["predit"], grouped["observe"], s=40, color=color,
                   edgecolor=SURFACE, linewidth=2, zorder=4)

        ax.set_title(label, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_axisbelow(True)
        ax.set_xlabel("xG moyen prédit")
        ax.xaxis.set_major_formatter(french(1))
        ax.yaxis.set_major_formatter(french(1))

    axes[0].set_ylabel("Part de buts observée")
    axes[0].text(0.50, 0.46, "calibration parfaite", color=MUTED, fontsize=9,
                 rotation=45, rotation_mode="anchor", ha="center", va="top")
    fig.suptitle(f"Calibration sur les {thousands(len(predictions))} tirs de test "
                 "(10 tranches, intervalles à 95 %)", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def log_loss_by_step(metrics: pd.DataFrame, path: Path) -> None:
    """Chaque feature ajoutée : log loss en validation croisée et sur le test."""
    steps = metrics.drop(index="xG officiel StatsBomb")
    reference = metrics.loc["xG officiel StatsBomb", "log_loss"]
    y = np.arange(len(steps))[::-1]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    series = [("log_loss_cv", "Validation croisée (train)", "#2a78d6"),
              ("log_loss", "Test (57 matchs)", "#eb6834")]
    for column, label, color in series:
        ax.plot(steps[column], y, color=color, zorder=2)
        ax.scatter(steps[column], y, s=40, color=color, edgecolor=SURFACE, linewidth=2,
                   zorder=3, label=label)

    ax.axvline(reference, color=MUTED, linewidth=1, zorder=1)
    ax.text(reference, -0.45, f"  xG StatsBomb (test) : {reference:.3f}".replace(".", ","),
            color=INK_SECONDARY, fontsize=9, va="center")

    ax.set_yticks(y, steps.index)
    ax.set_ylim(-0.6, len(steps) - 0.1)
    ax.set_xlabel("Log loss (plus bas = mieux)")
    ax.xaxis.set_major_formatter(french(3))
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", frameon=False, labelcolor=INK_SECONDARY)
    ax.set_title("Chaque feature ajoutée fait baisser le log loss de la logistique",
                 loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def draw_shot_map(player_shots: pd.DataFrame, title: str, highlight=None):
    """Tirs d'un joueur : taille = xG, rempli = but. highlight : index d'un tir à entourer."""
    goals = player_shots[player_shots["is_goal"] == 1]
    misses = player_shots[player_shots["is_goal"] == 0]
    color = "#2a78d6"

    # On coupe le bas du demi-terrain vide, sans jamais cacher un tir lointain.
    pad_bottom = -min(18.0, max(0.0, player_shots["x"].min() - 64))
    pitch = VerticalPitch(pitch_type="statsbomb", half=True, pitch_color=SURFACE,
                          line_color=BASELINE, linewidth=1, pad_bottom=pad_bottom)
    fig, ax = pitch.draw(figsize=(7, 6))

    def size(xg):
        return 60 + 1400 * np.asarray(xg)

    pitch.scatter(misses["x"], misses["y"], s=size(misses["xg"]), facecolor="none",
                  edgecolor=MUTED, linewidth=1.5, ax=ax, zorder=2)
    pitch.scatter(goals["x"], goals["y"], s=size(goals["xg"]), color=color,
                  edgecolor=SURFACE, linewidth=2, ax=ax, zorder=3)
    if highlight is not None:
        shot = player_shots.loc[highlight]
        pitch.scatter(shot["x"], shot["y"], s=size(shot["xg"]) * 2.2 + 150, facecolor="none",
                      edgecolor=INK, linewidth=2.5, ax=ax, zorder=4)

    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=9, color=color, label="But"),
        Line2D([], [], marker="o", linestyle="", markersize=9, markerfacecolor="none",
               markeredgecolor=MUTED, label="Tir non marqué"),
    ] + [
        Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=MUTED,
               markersize=np.sqrt(size(v)), label=f"xG {v:.2f}".replace(".", ","))
        for v in (0.05, 0.2, 0.5)
    ]
    ax.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=9,
              labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, 0.0), handletextpad=0.9,
              columnspacing=1.8, borderaxespad=0)

    n_goals, total_xg = int(player_shots["is_goal"].sum()), player_shots["xg"].sum()
    ax.set_title(f"{title}\n{len(player_shots)} tirs, {n_goals} buts pour "
                 f"{total_xg:.1f} xG".replace(".", ","), loc="left", fontsize=12, color=INK)
    return fig


def shot_map(shots: pd.DataFrame, player: str, title: str, path: Path) -> None:
    fig = draw_shot_map(shots[shots["player"] == player], title)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def draw_explanation(shot: pd.Series, top: int = 7):
    """Du tir moyen à l'xG du tir : ce que chaque feature ajoute ou retire."""
    contributions = pd.Series({f: shot[f"contrib__{f}"] for f in NUMERIC + CATEGORICAL})
    order = contributions.abs().sort_values(ascending=False).index
    kept, rest = order[:top], order[top:]

    steps = [(f"{LABELS[f]} : {describe(f, shot[f])}", contributions[f]) for f in kept]
    if len(rest):
        steps.append((f"{len(rest)} autres features", contributions[rest].sum()))

    logit = shot["base_logit"]
    rows = [("Tir moyen", None, sigmoid(logit), sigmoid(logit))]
    for label, delta in steps:
        before, logit = sigmoid(logit), logit + delta
        rows.append((label, delta, before, sigmoid(logit)))
    rows.append(("xG de ce tir", None, sigmoid(logit), sigmoid(logit)))

    fig, ax = plt.subplots(figsize=(5.6, 0.42 * len(rows) + 0.9))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    y = np.arange(len(rows))[::-1]
    right = max(max(r[2], r[3]) for r in rows)

    for yi, (label, delta, before, after) in zip(y, rows):
        if delta is None:
            ax.scatter([after], [yi], s=60, color=INK, zorder=3)
            ax.text(after + right * 0.02, yi, number(after, 2), va="center", color=INK, fontsize=12,
                    fontweight="bold")
            continue
        color = INCREASE if after >= before else DECREASE
        ax.barh(yi, after - before, left=before, height=0.55, color=color, zorder=2)
        sign = "+" if after >= before else "−"
        ax.text(max(before, after) + right * 0.02, yi, f"{sign}{number(abs(after - before), 3)}",
                va="center", color=INK_SECONDARY, fontsize=11)

    ax.set_yticks(y, [r[0] for r in rows], color=INK, fontsize=12)
    ax.set_xlim(0, right * 1.18)
    ax.xaxis.set_major_formatter(french(1 if right > 0.5 else 2))  # graduations 0,05 sous 0,5
    ax.set_xlabel("Probabilité de but", color=INK_SECONDARY, fontsize=11)
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=MUTED, labelcolor=INK_SECONDARY, labelsize=11)
    fig.tight_layout()
    return fig


def xg_map(shots: pd.DataFrame, path: Path, min_shots: int = 10) -> None:
    """xG moyen par zone. Zones de moins de 10 tirs laissées vides : trop peu pour une moyenne."""
    pitch = VerticalPitch(pitch_type="statsbomb", half=True, pitch_color=SURFACE,
                          line_color=BASELINE, linewidth=1, line_zorder=2, pad_bottom=-18)
    fig, ax = pitch.draw(figsize=(7, 6))

    stats = pitch.bin_statistic(shots["x"], shots["y"], values=shots["xg"], statistic="mean",
                                bins=(24, 16))
    counts = pitch.bin_statistic(shots["x"], shots["y"], statistic="count", bins=(24, 16))
    stats["statistic"] = np.where(counts["statistic"] >= min_shots, stats["statistic"], np.nan)

    cmap = LinearSegmentedColormap.from_list("bleu", BLUE_RAMP)
    mesh = pitch.heatmap(stats, ax=ax, cmap=cmap, vmin=0, vmax=0.4, edgecolor=SURFACE, linewidth=1)
    colorbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02,
                            shrink=0.6)
    colorbar.set_label("xG moyen des tirs de la zone", color=INK_SECONDARY)
    colorbar.outline.set_visible(False)
    colorbar.ax.xaxis.set_major_formatter(french(2))
    colorbar.ax.tick_params(color=MUTED, labelcolor=INK_SECONDARY)

    ax.set_title(f"Où l'on marque : xG moyen par zone ({thousands(len(shots))} tirs hors penalty)",
                 loc="left", fontsize=12, color=INK)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)

    predictions = pd.read_csv(REPORTS / "predictions_test.csv")
    calibration(predictions, FIGURES / "calibration.png")

    metrics = pd.read_csv(REPORTS / "metrics_logistique.csv", index_col="modele")
    log_loss_by_step(metrics, FIGURES / "log_loss_par_etape.png")

    df = pd.read_csv(DATA)
    shots = df[df["shot_type"] != "Penalty"].copy()
    shots = pd.concat([shots, out_of_fold_explanations(shots)], axis=1)
    shot_map(shots, "Kylian Mbappé Lottin", "Kylian Mbappé, Coupes du monde 2018 et 2022",
             FIGURES / "carte_tirs_mbappe.png")
    xg_map(shots, FIGURES / "carte_xg.png")

    # Un but de Mbappé à l'xG modeste : des features qui poussent dans les deux sens.
    goals = shots[(shots["player"] == "Kylian Mbappé Lottin") & (shots["is_goal"] == 1)]
    shot = goals.loc[(goals["xg"] - 0.15).abs().idxmin()]
    fig = draw_explanation(shot)
    fig.savefig(FIGURES / "explication_tir.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"tir expliqué : {shot['competition']}, {int(shot['minute']) + 1}e minute, xG {shot['xg']:.2f}")

    print(f"figures écrites dans {FIGURES}")


if __name__ == "__main__":
    main()
