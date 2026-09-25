# Modèle d'expected goals (xG)

Estimer la proba que le tir d'un match de football termine au fond du but.
Comparaison avec StatsBomb.

Les notions et les résultats sont expliqués pas à pas dans
[`rapport/rapport.txt`](rapport/rapport.txt). Ce README sert à faire tourner le code.

## Ce qu'il y a dans chaque fichier

| Fichier | Rôle |
| --- | --- |
| `src/statsbomb.py` | Télécharge les JSON du dépôt open-data et les met en cache dans `data/raw/` |
| `src/features.py` | Distance, angle entre les poteaux, défenseurs dans le triangle de tir |
| `src/dataset.py` | Parcourt les matchs, aplatit chaque tir en une ligne, écrit `data/shots.csv` |
| `src/train.py` | Logistique enrichie feature par feature, métriques, calibration, comparaison à l'xG StatsBomb |
| `src/boosting.py` | Gradient boosting réglé par validation croisée, bootstrap par match contre la logistique et StatsBomb |
| `tests/test_features.py` | Géométrie : une erreur de signe fausse tout sans lever d'exception |

## Démarrage

```bash
cd ~/Documents/projects/01-xg-model
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python tests/test_features.py          # 9 tests, doit passer avant tout le reste
python src/dataset.py --limit 5        # essai rapide : 5 matchs par compétition
python src/dataset.py                  # le jeu complet, compter une dizaine de minutes
python src/train.py                    # logistique, feature par feature
python src/boosting.py                 # gradient boosting, une à deux minutes
```

Le premier téléchargement complet prend un moment : un appel HTTP par match.
Tout est mis en cache, la deuxième exécution est instantanée.

## Les trois décisions de méthode

**Le découpage train/test se fait par match, pas par tir.** Deux tirs du même
match partagent le contexte. Les séparer ferait fuiter de l'information et
gonflerait le score sans que rien ne le signale.

**L'accuracy n'apparaît nulle part.** Environ 10 % des tirs sont des buts, donc
un modèle qui répond « jamais but » afficherait 90 %. On mesure le log loss et
le score de Brier, qui punissent la confiance mal placée.

**La calibration est le vrai résultat.** Parmi les tirs auxquels le modèle
donne 0,30, il doit y avoir environ 30 % de buts. Un modèle d'xG n'est utile
que dans la mesure où il est calibré — c'est la première figure du README final.

## Le juge de paix

StatsBomb livre son propre xG (`shot_statsbomb_xg`) à côté de chaque tir.
`train.py` évalue les deux sur le même jeu de test. La baseline à deux
variables ne le battra pas, et c'est le point : l'écart mesure ce que
distance et angle ne capturent pas, et justifie les features suivantes.

## Suite de la semaine

- [x] Chargement des données et mise en cache
- [x] Distance, angle, défenseurs dans le triangle (gardien compté comme défenseur)
- [x] Baseline logistique, métriques, calibration
- [x] Features catégorielles : partie du corps, situation de jeu, première intention
- [x] Gradient boosting, comparé à la baseline sur les mêmes métriques
- [ ] Carte de tirs avec mplsoccer
- [ ] Application Streamlit : carte de tirs, sur/sous-performance des joueurs, SHAP
- [ ] README final avec la courbe de calibration en première image

## Données

[StatsBomb Open Data](https://github.com/statsbomb/open-data), libres d'usage
sous réserve de citer StatsBomb. Aucune donnée n'est versionnée dans ce dépôt.
