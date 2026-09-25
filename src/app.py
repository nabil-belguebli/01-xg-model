"""Application : joueurs au-dessus ou en dessous de leur xG, cartes de tirs, explication d'un tir.

    python src/app_data.py      # une fois, après dataset.py
    streamlit run src/app.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from figures import draw_explanation, draw_shot_map, fr, number
from overperformance import expected_false_alarms, player_table

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "data" / "app_shots.csv"
REPO = "https://github.com/nabil-belguebli/01-xg-model"

@st.cache_data
def load_shots() -> pd.DataFrame:
    return pd.read_csv(SHOTS)


@st.cache_data
def players(shots: pd.DataFrame, min_shots: int) -> tuple[pd.DataFrame, float]:
    table = player_table(shots, min_shots)
    return table, expected_false_alarms(shots, table)


def tab_players(shots: pd.DataFrame):
    min_shots = st.slider("Nombre minimum de tirs", 5, 40, 15)
    table, expected = players(shots, min_shots)
    if table.empty:
        st.info("Aucun joueur avec autant de tirs dans cette sélection.")
        return

    below = int((table["proba_hasard"] < 0.05).sum())
    left, middle, right = st.columns(3)
    left.metric("Joueurs affichés", len(table))
    middle.metric("Probabilité de hasard sous 5 %", below)
    right.metric("Attendus par pur hasard", number(expected))

    st.dataframe(
        table.assign(proba_hasard=table["proba_hasard"] * 100),
        hide_index=True,
        width="stretch",
        column_config={
            "joueur": st.column_config.TextColumn("Joueur", width="medium"),
            "equipe": st.column_config.TextColumn("Équipe", width="small"),
            "tirs": st.column_config.NumberColumn("Tirs", width="small"),
            "buts": st.column_config.NumberColumn("Buts", width="small"),
            "xg": st.column_config.NumberColumn("xG", format="%.1f", width="small"),
            "ecart": st.column_config.NumberColumn("Buts − xG", format="%+.1f", width="small"),
            "proba_hasard": st.column_config.NumberColumn(
                "Hasard", format="%.1f %%", width="small",
                help="Probabilité qu'un joueur quelconque, avec exactement ces occasions, "
                     "s'écarte autant de son xG par pure chance."),
        },
    )

    with st.expander("Comment lire ce tableau"):
        st.markdown(f"""
**Buts − xG** : buts marqués (hors penalty) moins buts attendus. Positif, le joueur a
marqué plus que ses occasions ne le laissaient prévoir.

**Probabilité de hasard** : on suppose que chaque tir est un tirage au sort dont la
probabilité de but est son xG, et on calcule la probabilité d'un écart au moins aussi
grand, dans le même sens. Plus elle est faible, moins la chance suffit à l'expliquer.

**Le piège des comparaisons multiples** : parmi {len(table)} joueurs, même si tous
n'avaient que de la chance, environ {number(expected)} auraient une probabilité sous
5 %. Ici, {below} y sont. Un joueur isolé sous 5 % ne prouve donc presque rien ;
c'est l'excès par rapport à ce nombre attendu qui serait un signal.
""")


def tab_shots(shots: pd.DataFrame):
    counts = shots["player"].value_counts()
    names = counts.index.tolist()
    default = names.index("Kylian Mbappé Lottin") if "Kylian Mbappé Lottin" in names else 0
    player = st.selectbox("Joueur", names, index=default,
                          format_func=lambda n: f"{n} ({counts[n]} tirs)")
    player_shots = shots[shots["player"] == player].sort_values("xg", ascending=False)

    listing = pd.DataFrame({
        "Compétition": player_shots["competition"],
        "Minute": player_shots["minute"] + 1,  # StatsBomb compte les minutes écoulées
        "Partie du corps": player_shots["body_part"].map(fr),
        "Passe décisive": player_shots["assist_type"],
        "xG": player_shots["xg"],
        "Résultat": np.where(player_shots["is_goal"] == 1, "But", "—"),
    }, index=player_shots.index)

    st.caption("Cliquez sur une ligne pour expliquer ce tir.")
    selection = st.dataframe(
        listing, hide_index=True, width="stretch", height=220,
        on_select="rerun", selection_mode="single-row",
        column_config={"xG": st.column_config.NumberColumn(format="%.2f")},
    )
    rows = selection.selection.rows
    chosen = player_shots.index[rows[0]] if rows else player_shots.index[0]
    shot = player_shots.loc[chosen]

    left, right = st.columns(2)
    with left:
        fig = draw_shot_map(player_shots, player, highlight=chosen)
        st.pyplot(fig)
        plt.close(fig)
    with right:
        result = "but" if shot["is_goal"] == 1 else "non marqué"
        st.markdown(f"**Tir entouré** : {shot['competition']}, {int(shot['minute']) + 1}e minute, "
                    f"{fr(shot['body_part']).lower()}, {result}. xG **{number(shot['xg'], 2)}**.")
        fig = draw_explanation(shot)
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "On part du tir moyen, puis chaque feature ajoute (bleu) ou retire (rouge) de la "
            "probabilité de but. Ce sont les valeurs SHAP du modèle logistique : calculées en "
            "log-cotes, converties en probabilité dans l'ordre affiché. Le modèle qui a prédit ce "
            "tir n'avait pas vu son match.")


def tab_model():
    st.markdown("""
Régression logistique sur 17 features : géométrie du tir, défenseurs, gardien, passe
décisive, phase de jeu. Apprise sur 5 583 tirs hors penalty (Coupes du monde 2018 et 2022,
Coupe du monde féminine 2019, Euro 2020). Chaque xG affiché est prédit par un modèle qui
n'a pas vu le match du tir.
""")
    metrics = ROOT / "reports" / "metrics_modeles.csv"
    if metrics.exists():
        table = pd.read_csv(metrics).rename(columns={"modele": "Modèle", "log_loss": "Log loss",
                                                     "brier": "Brier", "roc_auc": "ROC AUC"})
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="%.3f") for c in ["Log loss", "Brier", "ROC AUC"]})
    calibration = ROOT / "figures" / "calibration.png"
    if calibration.exists():
        st.image(str(calibration))
    st.markdown(f"Méthode et notions expliquées pas à pas : [rapport]({REPO}/blob/main/rapport/rapport.txt). "
                "Données : [StatsBomb Open Data](https://github.com/statsbomb/open-data).")


def main():
    st.set_page_config(page_title="Modèle d'xG", layout="wide")
    st.title("Modèle d'expected goals")
    st.caption("Probabilité qu'un tir finisse au fond, estimée sur 5 583 tirs de Coupe du monde et d'Euro.")

    if not SHOTS.exists():
        st.error("Données manquantes. Lancez d'abord `python src/dataset.py` puis `python src/app_data.py`.")
        st.stop()

    shots = load_shots()
    competitions = sorted(shots["competition"].unique())
    chosen = st.sidebar.multiselect("Compétitions", competitions, default=competitions)
    shots = shots[shots["competition"].isin(chosen)]
    if shots.empty:
        st.info("Choisissez au moins une compétition.")
        st.stop()

    players_tab, shots_tab, model_tab = st.tabs(["Joueurs", "Tirs d'un joueur", "Le modèle"])
    with players_tab:
        tab_players(shots)
    with shots_tab:
        tab_shots(shots)
    with model_tab:
        tab_model()


main()
