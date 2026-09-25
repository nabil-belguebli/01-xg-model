// Rapport du projet xG. Compiler : python rapport/build.py
// Chiffres issus de reports/*.csv ; figures de figures/ (python src/figures.py).

#let accent = rgb("#2a78d6")
#let ink-secondary = rgb("#52514e")
#let wash = rgb("#eef4fc")
#let rule = rgb("#c3c2b7")

#set document(title: "Modèle d'expected goals", author: "Nabil Belguebli")
#set page(paper: "a4", margin: (x: 2.4cm, top: 2.4cm, bottom: 2.6cm), numbering: "1",
  number-align: center)
#set text(font: "Libertinus Serif", size: 11pt, lang: "fr")
#set par(justify: true, leading: 0.68em, spacing: 1.1em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => {
  v(1.2em)
  set text(size: 17pt)
  block(below: 0.9em)[#if it.numbering != none [#text(fill: accent)[#counter(heading).display()] #h(0.3em)]#it.body]
}
#show heading.where(level: 2): it => {
  v(0.5em)
  set text(size: 12.5pt)
  block(below: 0.7em)[#if it.numbering != none [#text(fill: accent)[#counter(heading).display()] #h(0.2em)]#it.body]
}
#show figure.caption: it => text(size: 9.5pt, fill: ink-secondary)[#it]
#show link: it => text(fill: accent)[#it]
#set table(stroke: none, inset: (x: 6pt, y: 4.5pt))
#show table: set block(breakable: false)
#set math.equation(numbering: none)

// À retenir : encadré en fin de section.
#let retenir(body) = block(fill: wash, inset: (x: 12pt, y: 10pt), radius: 4pt, width: 100%,
  above: 1.2em, below: 1.2em)[#text(fill: accent, weight: "bold")[À retenir.] #body]

// Tableau de chiffres : filets en haut, sous l'en-tête et en bas.
#let chiffres(columns, align, header, ..rows) = table(
  columns: columns, align: align,
  table.hline(stroke: 0.8pt + rule),
  table.header(..header.map(h => text(weight: "bold")[#h])),
  table.hline(stroke: 0.5pt + rule),
  ..rows,
  table.hline(stroke: 0.8pt + rule),
)

// ---------------------------------------------------------------- Titre

#align(center)[
  #v(2.5cm)
  #text(size: 26pt, weight: "bold")[Modèle d'expected goals]
  #v(0.3em)
  #text(size: 14pt, fill: ink-secondary)[Estimer la probabilité qu'un tir finisse au fond, \
    et comprendre ce qui la fait varier]
  #v(1.2em)
  Nabil Belguebli · septembre 2026
  #v(0.2em)
  #text(size: 10pt, fill: ink-secondary)[
    Code : #link("https://github.com/nabil-belguebli/01-xg-model")[github.com/nabil-belguebli/01-xg-model] \
    Données : StatsBomb Open Data]
]

#v(1.5cm)

#block(stroke: (left: 2pt + accent), inset: (left: 14pt, y: 4pt))[
  #text(weight: "bold")[En bref.] Sur 5 583 tirs de Coupe du monde et d'Euro, on construit un
  modèle d'expected goals (xG) et on le compare à celui de StatsBomb, une entreprise
  d'analyse de données de football. Quatre résultats :

  - Une régression logistique à 17 features atteint un log loss de 0,256, contre 0,280
    pour la distance et l'angle seuls et 0,245 pour StatsBomb : 70 % de l'écart est comblé.
  - Un gradient boosting ne fait pas mieux que la logistique. Ce sont les features qui
    comptent, pas la puissance du modèle.
  - Les trois modèles sont bien calibrés : leurs probabilités tiennent parole.
  - Les joueurs qui marquent plus que leur xG sont à peine plus nombreux que ce que le
    hasard seul produirait : sur ces tournois, la sur-performance est surtout de la chance.

  Ce document explique chaque étape et chaque notion sans prérequis en statistiques ni en
  programmation. Pour faire tourner le code, voir le README du dépôt.
]

#pagebreak()
#outline(title: [Sommaire], indent: auto, depth: 2)
#pagebreak()

// ---------------------------------------------------------------- 1

= La question

Un attaquant tire depuis le point de penalty, un autre depuis 30 mètres à l'angle de la
surface. Les deux ratent. Ont-ils raté la même chose ?

Évidemment non : le premier avait une vraie occasion, le second presque aucune. Compter
les buts ne voit pas cette différence ; les _expected goals_ (xG), si.

L'xG d'un tir est la probabilité qu'il finisse au fond, estimée à partir de tirs passés
qui lui ressemblent. Un xG de 0,12 veut dire : sur 100 tirs semblables à celui-ci,
environ 12 sont des buts.

Additionnés, les xG disent ce qu'une équipe ou un joueur « aurait dû » marquer. Un joueur
à 10 xG et 14 buts a marqué 4 buts de plus que la qualité de ses occasions ne le laissait
attendre : il sur-performe. Talent ou chance ? C'est la question qui rend l'xG utile aux
clubs, aux recruteurs et aux analystes.

Le projet a trois objectifs :
+ construire un modèle d'xG soi-même, et le comparer honnêtement à celui de StatsBomb ;
+ comprendre d'où vient l'écart entre les deux ;
+ utiliser le modèle pour répondre à la question « talent ou chance ? » et pour expliquer
  chaque prédiction.

// ---------------------------------------------------------------- 2

= Les données

StatsBomb publie gratuitement une partie de ses données. Pour chaque match, on dispose de
la liste de tous les événements : passes, dribbles, tirs… Pour un tir, on connaît
notamment :

- la position du tireur sur le terrain ;
- la partie du corps (pied droit, pied gauche, tête) ;
- le type de tir (jeu courant, coup franc, penalty) et la technique (frappe normale,
  volée, lob…) ;
- si le tireur était sous pression, s'il a tiré en première intention ;
- le _freeze frame_ : la position de tous les joueurs visibles à l'instant du tir ;
- la passe qui a amené le tir, si elle existe ;
- le résultat, but ou non ;
- l'xG calculé par StatsBomb, qui sert de référence à battre.

Quatre compétitions sont utilisées : les Coupes du monde 2018 et 2022, la Coupe du monde
féminine 2019 et l'Euro 2020. Au total : 231 matchs et 5 803 tirs, dont 676 buts
(11,6 %).

Les 220 penalties sont retirés. Un penalty est un tir à part : même position, aucun
défenseur, environ 75 % de réussite. Le mélanger aux autres brouillerait le modèle, et
son xG est connu d'avance. Il reste *5 583 tirs, dont 9,5 % de buts*.

Le terrain StatsBomb mesure 120 × 80 unités, à peu près des yards (une unité vaut environ
0,9 mètre). Le but est en $x = 120$, entre les poteaux $y = 36$ et $y = 44$.

// ---------------------------------------------------------------- 3

= Des tirs aux nombres : les features

Un modèle ne « voit » pas un tir : il voit une liste de nombres qui le décrivent. On
appelle ces nombres des _features_ (variables explicatives). Les choisir et les calculer
compte souvent plus que le choix du modèle ; la section 6 le confirmera.

== Distance et angle

La *distance* est celle qui sépare le tireur du centre de la ligne de but. Plus on est
loin, moins on marque.

L'*angle de tir* est l'angle sous lequel le tireur voit l'ouverture du but. Face au but à
11 mètres, il voit une large ouverture ; depuis le coin de la surface, à la même distance,
il n'en voit qu'une fente. La distance seule ne fait pas cette différence.

On le calcule par la loi des cosinus dans le triangle formé par le tireur et les deux
poteaux. Si $a$ et $b$ sont les distances du tireur à chaque poteau et $w = 8$ la largeur
du but :
$ cos theta = (a^2 + b^2 - w^2) / (2 a b) $

== Défenseurs dans le triangle de tir

Grâce au freeze frame, on compte les adversaires situés dans ce même triangle
tireur–poteau–poteau, c'est-à-dire entre le ballon et le but. Le gardien compte comme un
défenseur. Pour savoir si un joueur est dans le triangle, on regarde de quel côté de
chaque arête il se trouve (signe d'un produit vectoriel) : s'il est du même côté des trois
arêtes, il est dedans.

Les données brutes montrent tout de suite que c'est important :

#align(center, chiffres((auto, auto, auto), (left, right, right),
  ([Défenseurs dans le triangle], [Tirs], [Part de buts]),
  [0 (but vide)], [142], [34 %],
  [1 (en général le gardien)], [2 549], [13 %],
  [2], [1 882], [6 %],
  [3], [619], [3 %],
))

D'autres features s'ajouteront en section 7 : position du gardien, défenseur le plus
proche, passe décisive, phase de jeu.

== Des tests pour la géométrie

Une erreur de signe dans ces calculs ne fait pas planter le programme : elle produit des
chiffres faux, en silence. Des tests automatiques vérifient donc la géométrie sur des cas
dont on connaît la réponse : le point de penalty est à 12 unités du but, l'angle est
maximal dans l'axe et quasi nul depuis la ligne de sortie, un défenseur placé derrière le
tireur n'est pas dans le triangle, etc. Le projet compte 21 tests.

// ---------------------------------------------------------------- 4

= Juger un modèle d'xG

C'est la partie la plus importante : un modèle mal évalué peut sembler excellent et être
inutile.

== Entraînement et test, découpés par match

On apprend sur 75 % des matchs (174 matchs, 4 221 tirs) et on évalue sur les 25 % restants
(57 matchs, 1 362 tirs), que le modèle n'a jamais vus. Évaluer un modèle sur ses données
d'apprentissage, c'est noter un élève sur les exercices dont il a appris le corrigé.

Le découpage se fait *par match*, pas par tir. Deux tirs d'un même match partagent un
contexte : mêmes équipes, même gardien, même météo. Si l'un sert à apprendre et l'autre à
tester, le modèle a déjà un indice sur le test. Cette _fuite d'information_ gonfle les
scores sans jamais provoquer d'erreur visible ; on s'en protège donc par construction.

== Pourquoi pas le pourcentage de bonnes réponses

Seuls 9,5 % des tirs sont des buts. Un modèle qui répondrait « jamais but » aurait 90,5 %
de bonnes réponses tout en étant inutile. Surtout, un modèle d'xG ne répond pas oui ou
non : il donne une probabilité. Il faut juger la qualité de ces probabilités. Trois
mesures s'en chargent, et une figure.

== Le log loss (plus bas = mieux)

Pour chaque tir, on regarde la probabilité que le modèle avait donnée à ce qui s'est
réellement passé, et on compte une pénalité de $-log("probabilité")$. Le log loss est la
moyenne de ces pénalités. Avec $y_i = 1$ pour un but, $0$ sinon, et $p_i$ l'xG prédit :
$ "log loss" = -1/N sum_(i=1)^N [ y_i log p_i + (1 - y_i) log(1 - p_i) ] $

Deux exemples : un but auquel le modèle donnait 0,80 coûte −log 0,80 = 0,22 ; un but
auquel il donnait 0,02 coûte −log 0,02 = 3,9. Le logarithme punit durement la confiance
mal placée : se tromper en étant sûr de soi coûte bien plus cher qu'hésiter.

*Repère* : un modèle qui donne à chaque tir le taux moyen de buts, sans rien regarder,
obtient 0,323 sur notre test. Tout modèle doit faire mieux pour avoir appris quelque chose.

== Le score de Brier (plus bas = mieux)

$ "Brier" = 1/N sum_(i=1)^N (p_i - y_i)^2 $

Même idée que le log loss, mais les grosses erreurs sont moins punies : un carré reste
borné, un logarithme non. Quand les deux mesures vont dans le même sens, la conclusion est
solide. Repère : 0,089 pour le modèle qui ne regarde rien.

== La ROC AUC (plus haut = mieux)

On prend au hasard un but et un tir raté : quelle est la probabilité que le modèle ait
donné un xG plus élevé au but ? C'est l'AUC. Elle vaut 0,5 pour un modèle qui tire à pile
ou face, 1 pour un classement parfait.

L'AUC ne juge que l'*ordre* des tirs, pas la justesse des probabilités : un modèle qui
diviserait toutes ses probabilités par 10 garderait la même AUC tout en étant faux. On ne
la regarde donc jamais seule.

== La calibration : le vrai résultat

Un modèle est *calibré* si ses probabilités tiennent parole : parmi les tirs auxquels il
donne environ 0,15, il doit y avoir environ 15 % de buts. Pour le vérifier, on trie les
tirs de test par xG prédit, on les coupe en 10 tranches de même taille, et on compare dans
chaque tranche l'xG moyen prédit à la part de buts réellement observée (@fig-calibration).

C'est essentiel pour un xG, parce qu'on additionne les probabilités. Un modèle qui
surestimerait tout de 20 % donnerait à chaque joueur un total faux, même en classant
parfaitement les tirs.

#retenir[On découpe par match, on juge des probabilités (log loss, Brier) et non des
réponses, on regarde l'ordre (AUC) sans s'y limiter, et la calibration passe avant tout.]

// ---------------------------------------------------------------- 5

= Premier modèle : la régression logistique

== Le principe

Chaque feature reçoit un poids, appelé *coefficient*. On calcule un score, puis la
fonction sigmoïde, en forme de S, le ramène entre 0 et 1 pour en faire une probabilité :
$ s = c_0 + c_1 x_1 + c_2 x_2 + dots.h + c_k x_k quad quad quad p = 1 / (1 + e^(-s)) $

Un score de 0 donne 0,5, un score très négatif donne presque 0. On appelle aussi $s$ la
_log-cote_ : c'est le logarithme de $p slash (1 - p)$.

_Entraîner_ le modèle, c'est chercher les coefficients qui minimisent le log loss sur les
tirs d'entraînement. Son grand atout : il est *lisible*. Chaque coefficient dit dans quel
sens et avec quelle force une feature pousse la probabilité.

== Préparer les features

*Standardisation.* La distance se compte en dizaines d'unités, l'angle en radians, la
première intention vaut 0 ou 1. Pour comparer les coefficients, on ramène chaque variable
à une moyenne de 0 et un écart-type de 1. Un coefficient se lit alors : « effet d'un
écart-type de plus ».

*One-hot encoding.* Un modèle ne sait pas multiplier « tête » par un nombre. On remplace
la colonne « partie du corps » par une colonne par valeur possible (tête, pied gauche,
pied droit…) contenant 1 ou 0. Chaque catégorie reçoit ainsi son propre coefficient.

*Catégories rares.* Il y a 3 corners directs et 1 tir depuis le rond central. Un
coefficient appris sur 3 tirs n'est que du bruit : sous 20 tirs, les catégories sont
regroupées dans une case « rare ».

== La baseline, puis les features une à une

On commence toujours par le modèle le plus simple possible, pour mesurer ce que chaque
complication rapporte : ici, la distance et l'angle seuls. Puis on ajoute les features
une à une, chaque ligne gardant celles des précédentes, et tout est évalué sur les mêmes
1 362 tirs de test.

#align(center, chiffres((auto, auto, auto, auto), (left, right, right, right),
  ([Modèle], [Log loss], [Brier], [AUC]),
  [Aucun modèle (taux moyen)], [0,323], [0,089], [0,500],
  [Distance + angle], [0,280], [0,080], [0,768],
  [\+ partie du corps], [0,272], [0,078], [0,788],
  [\+ type de tir et technique], [0,268], [0,077], [0,797],
  [\+ première intention, pression], [0,265], [0,076], [0,806],
  [\+ défenseurs dans le triangle], [0,262], [0,074], [0,803],
  table.hline(stroke: 0.3pt + rule),
  [_xG StatsBomb (référence)_], [_0,245_], [_0,069_], [_0,833_],
))

La baseline fait déjà nettement mieux que le modèle qui ne regarde rien, mais reste loin
de StatsBomb. Chaque ajout améliore le log loss.

La dernière ligne améliore le log loss mais baisse légèrement l'AUC : les probabilités
deviennent plus justes sans que l'ordre des tirs change vraiment. C'est l'illustration
de la différence entre les deux mesures.

== Le piège de la première intention

Dans les données brutes, les tirs en première intention sont marqués 13 % du temps, contre
8 % pour les autres. Pourtant, le modèle leur donne un coefficient quasi nul (+0,05).
Pourquoi ? Ce sont surtout des reprises de centre, donc des tirs pris plus près du but :
une fois la distance connue, savoir que le tir était en première intention n'apprend plus
rien.

C'est un *facteur de confusion* : une variable semble avoir un effet alors qu'elle ne fait
que suivre une autre. Comparer deux pourcentages ne permet pas de le voir ; un modèle qui
considère toutes les variables en même temps, si.

#retenir[On part d'une baseline simple et on mesure l'apport de chaque ajout. Un effet
visible dans les données brutes peut disparaître une fois les autres variables prises en
compte : c'est un facteur de confusion.]

// ---------------------------------------------------------------- 6

= Deuxième modèle : le gradient boosting

== Le principe

Un *arbre de décision* pose des questions successives (« distance < 12 ? puis tête ?
puis plus d'un défenseur ? ») et donne une probabilité au bout de chaque chemin. Un arbre
seul est grossier. Le *gradient boosting* en empile des dizaines ou des centaines : chaque
nouvel arbre est entraîné à corriger les erreurs laissées par les précédents, et la
prédiction finale est la somme de leurs contributions.

Son avantage sur la logistique : il capte les *interactions* et les effets non linéaires.
Une tête peut être dangereuse de près et inutile de loin ; la logistique, qui additionne
des effets séparés, ne peut pas l'exprimer. Ses défauts : il est moins lisible, et il peut
apprendre par cœur le bruit des données d'entraînement. C'est le *surapprentissage*.

On lui donne exactement les mêmes features que la logistique, pour que seule la nature du
modèle change.

== Régler un modèle sans tricher : la validation croisée

Le gradient boosting a des réglages, les _hyperparamètres_ : nombre d'arbres, taille de
chaque arbre, vitesse d'apprentissage, nombre minimum de tirs par feuille. Trop gros ou
trop nombreux, les arbres apprennent le bruit.

Pour choisir, on ne peut pas regarder le jeu de test : on le « consommerait », et son
score ne serait plus une mesure honnête. On découpe donc le jeu d'entraînement en 5 parts,
par match toujours, on apprend sur 4 et on mesure sur la 5#super[e], cinq fois en tournant.
C'est la *validation croisée*. On essaie 144 combinaisons de réglages et on garde celle
qui a le meilleur log loss moyen.

Quand le meilleur réglage tombe au bord de la grille essayée, l'optimum est peut-être
au-delà : c'est arrivé deux fois, et on a chaque fois élargi la grille avant de conclure.
Le réglage final : 200 _souches_, des arbres à une seule question, avec au moins 50 tirs
par feuille. Un modèle très simple : avec 4 200 tirs dont 400 buts, il n'y a pas de quoi
nourrir un modèle complexe.

== Vrai écart ou hasard ? Le bootstrap

Le test ne compte que 57 matchs. Avec 57 autres matchs, les chiffres auraient un peu
bougé. Comment savoir si un écart de 0,001 entre deux modèles est réel ?

On simule d'autres jeux de test possibles : on tire 57 matchs au hasard parmi les 57 du
test, *avec remise* (certains reviennent deux fois, d'autres pas du tout), et on recalcule
l'écart entre deux modèles. On recommence 2 000 fois. On tire des matchs entiers et non des
tirs, pour la même raison qu'en section 4.1. Les 95 % d'écarts les plus centraux forment un
*intervalle de confiance* : s'il contient 0, on ne peut pas dire lequel des deux modèles
est meilleur.

#align(center, chiffres((auto, auto, auto), (left, right, center),
  ([Écart de log loss], [Observé], [Intervalle à 95 %]),
  [Logistique 17 features − baseline], [−0,025], [\[−0,036 ; −0,014\]],
  [Gradient boosting − logistique], [−0,001], [\[−0,006 ; +0,005\]],
  [Gradient boosting − StatsBomb], [+0,010], [\[+0,000 ; +0,019\]],
))

Lecture : les features ajoutées apportent un gain certain ; boosting et logistique sont à
égalité ; StatsBomb reste devant, de peu mais probablement pour de vrai.

== Ce que ça nous apprend

Changer de modèle n'a rien apporté ; ajouter des features, beaucoup. Les souches choisies
par la validation croisée le confirment : un empilement d'arbres à une seule question ne
peut pas combiner deux features, il additionne des effets séparés comme la logistique. Les
interactions n'apportent donc rien de mesurable ici. À information égale, la logistique a
en plus l'avantage d'être lisible : c'est elle qu'utilise la suite du projet.

#retenir[Les réglages se choisissent en validation croisée, jamais sur le test. Un écart
entre deux modèles ne compte que si son intervalle de confiance exclut 0.]

// ---------------------------------------------------------------- 7

= Donner plus d'information au modèle

Puisque le levier, ce sont les features, on en ajoute, toutes tirées des données
StatsBomb.

== Les nouvelles features

- *Le gardien*, d'après le freeze frame : sa distance à son but (est-il sorti de sa
  ligne ?) et sa distance au tireur (face-à-face ?).
- *Le défenseur le plus proche*, gardien exclu : le tireur a-t-il de l'espace ?
- *La passe décisive* : son type (en profondeur, en retrait, centre, coup de pied arrêté,
  autre passe, ou aucune) et sa hauteur (au sol, basse, haute).
- *Les annotations des analystes* : face-à-face avec le gardien, but vide, tir après un
  duel aérien gagné.
- *La phase de jeu* : attaque placée, contre-attaque, après un corner, une touche…

Le gardien est visible pour tous les tirs sauf 2 ; pour ceux-là, ses distances sont
remplacées par la valeur médiane. C'est une _imputation_ : la logistique ne sait pas
calculer avec une case vide.

#align(center, chiffres((auto, auto, auto), (left, right, right),
  ([Situation], [Tirs], [Part de buts]),
  [Passe décisive en profondeur], [154], [34 %],
  [Passe en retrait], [112], [18 %],
  [Centre], [862], [13 %],
  [Autre passe], [2 462], [7 %],
  [Gardien à plus de 10 unités de son but], [89], [34 %],
  [Contre-attaque], [228], [17 %],
  [Face-à-face], [219], [26 %],
  [But vide], [60], [73 %],
))

== Ne pas souffler la réponse au modèle

Certaines informations du fichier décrivent l'issue du tir, pas la situation au moment de
la frappe. `goal_assist` marque une passe décisive… seulement si le tir a fini au fond ;
`end_location` dit où le ballon a terminé ; `deflected` raconte ce qui s'est passé après.
Les donner au modèle, ce serait lui souffler la réponse : il aurait des scores
spectaculaires en test et serait inutilisable en vrai, puisqu'au moment de la frappe ces
informations n'existent pas. C'est une autre forme de fuite.

La règle : n'utiliser que ce qui est connu à l'instant du tir. Un test automatique vérifie
que la passe décisive est décrite de la même façon, que le tir soit un but ou non.

== Le piège du jeu de test

En ajoutant ces features une à une et en regardant le seul score de test, deux étapes
semblaient faire *reculer* le modèle. Étrange, vu la force des signaux bruts. On a donc
mesuré chaque étape en validation croisée sur le jeu d'entraînement, c'est-à-dire en
moyenne sur 5 découpages au lieu d'un seul jeu de 57 matchs (@fig-etapes).

#figure(image("../figures/log_loss_par_etape.png", width: 88%),
  caption: [Log loss de la logistique à chaque ajout de features. En validation croisée
  (bleu), chaque étape améliore le modèle ; sur le seul test (orange), les deux dernières
  semblent le dégrader : c'est le bruit d'un jeu de 57 matchs.]) <fig-etapes>

En validation croisée, chaque étape améliore bien le modèle. Les reculs vus sur le test
étaient du bruit : 57 matchs ne suffisent pas à départager des écarts de 0,002. Si l'on
choisissait les features en regardant le test, on finirait par ajuster le modèle aux
hasards de ces 57 matchs, et le score de test ne serait plus une mesure honnête.

== La régularisation

Le modèle compte maintenant 44 coefficients pour environ 400 buts d'entraînement. Avec
autant de paramètres, il risque de commencer à apprendre le bruit. La *régularisation* est
un frein : elle pénalise les coefficients trop grands et les ramène vers 0, sauf si les
données les justifient clairement. Un réglage noté $C$ dose ce frein : plus $C$ est petit,
plus le frein est fort. La validation croisée a retenu $C = "0,1"$, un frein modéré. Tous
les chiffres de logistique de ce rapport utilisent ce réglage.

== Ce que disent les coefficients

#align(center, chiffres((auto, auto, auto), (left, right, left),
  ([Feature (standardisée)], [Coefficient], [Lecture]),
  [Angle de tir], [+0,57], [la plus forte],
  [Passe en profondeur], [+0,51], [la plus forte des nouvelles],
  [Distance au but], [−0,33], [],
  [Défenseurs dans le triangle], [−0,30], [],
  [Gardien – tireur], [−0,28], [gardien proche, moins de buts],
  [Défenseur le plus proche], [+0,27], [de l'espace aide],
  [Après un corner], [−0,33], [têtes dans la foule],
  [Duel aérien gagné], [−0,22], [],
  [Gardien – but], [+0,13], [gardien sorti, but plus ouvert],
  [Première intention], [+0,05], [quasi nul, voir 5.4],
  [But vide], [0,00], [!],
))

Le but vide a un coefficient nul alors que 73 % de ces tirs finissent au fond. C'est le
même mécanisme que la première intention : un tir « but vide » est pris en moyenne à
4,5 unités du but, avec 0,4 défenseur dans le triangle et un gardien sorti. Les autres
features le décrivent déjà entièrement ; l'étiquette n'apporte rien de plus.

== Résultats

#align(center, chiffres((auto, auto, auto, auto), (left, right, right, right),
  ([Modèle], [Log loss], [Brier], [AUC]),
  [Distance + angle], [0,280], [0,080], [0,768],
  [Logistique, 17 features], [0,256], [0,073], [0,816],
  [Gradient boosting, 17 features], [0,255], [0,073], [0,819],
  [xG StatsBomb], [0,245], [0,069], [0,833],
))

On comble environ 70 % de l'écart entre la baseline et StatsBomb. Ce qui manque encore :
la hauteur du ballon à la frappe, la position de tous les défenseurs et pas seulement du
plus proche, et surtout beaucoup plus de données — StatsBomb entraîne son modèle sur des
centaines de milliers de tirs, nous sur 4 200.

#retenir[N'utiliser que l'information disponible au moment du tir. Choisir les features
et les réglages en validation croisée. La régularisation empêche un modèle à beaucoup de
paramètres de croire au bruit.]

// ---------------------------------------------------------------- 8

= Les modèles tiennent-ils parole ?

#figure(image("../figures/calibration.png", width: 100%),
  caption: [Calibration sur les 1 362 tirs de test. Chaque point est une tranche de
  136 tirs ; barres : intervalles à 95 %. Un modèle parfaitement calibré a ses points sur
  la diagonale.]) <fig-calibration>

La @fig-calibration dessine la calibration des trois modèles. Les barres verticales sont
des intervalles à 95 % : même avec un modèle parfait, une tranche de 136 tirs ne tombe
jamais exactement sur la diagonale, par simple hasard, comme 10 lancers de pièce ne
donnent pas toujours 5 piles. Pour une part de buts observée $hat(q)$ sur $n$ tirs :
$ hat(q) plus.minus "1,96" sqrt((hat(q) thin (1 - hat(q))) / n) $

Tant que la diagonale passe dans la barre, l'écart n'est pas un défaut du modèle. Les
trois modèles sont bien calibrés. Tous sous-estiment un peu les plus grosses occasions
(xG moyen 0,38 pour 44 % de buts dans la dernière tranche de la logistique), mais les
barres touchent la diagonale ou en sont très proches. Sur l'ensemble des 5 583 tirs, la
somme des xG vaut 528,7 pour 529 buts.

== Un xG pour chaque tir : les prédictions hors-pli

Pour les cartes et les bilans de joueurs, il faut un xG pour chacun des 5 583 tirs, pas
seulement pour les 1 362 du test. Mais un modèle ne doit jamais juger un tir qu'il a vu
pendant son entraînement. Solution : on découpe les matchs en 5 groupes, on entraîne sur 4
et on prédit le 5#super[e], cinq fois. Chaque tir reçoit l'xG d'un modèle qui n'a jamais vu
son match. Ce sont des prédictions *hors-pli* (_out of fold_) : le mécanisme de la
validation croisée, utilisé pour produire des prédictions au lieu d'un score.

#figure(image("../figures/carte_xg.png", width: 70%),
  caption: [xG moyen par zone du terrain. Les zones de moins de 10 tirs restent vides : une
  moyenne sur 3 tirs ne veut rien dire.]) <fig-carte-xg>

La @fig-carte-xg montre où l'on marque : tout se joue dans une zone étroite face au but.
Devant la cage, l'xG moyen dépasse 0,35 ; à l'entrée de la surface, il tombe sous 0,10 ;
de loin, il reste sous 0,05.

// ---------------------------------------------------------------- 9

= Talent ou chance ?

== Le cas Mbappé

#figure(image("../figures/carte_tirs_mbappe.png", width: 72%),
  caption: [Les 51 tirs hors penalty de Kylian Mbappé aux Coupes du monde 2018 et 2022.
  Taille : xG du tir ; cercles pleins : buts.]) <fig-mbappe>

Kylian Mbappé a marqué 10 buts hors penalty pour 6,1 xG (StatsBomb donne 5,9) : environ
4 buts de plus que ses occasions ne le laissaient attendre (@fig-mbappe). Talent ou
chance ? On peut le chiffrer.

Supposons que chaque tir soit un tirage au sort dont la probabilité de but est son xG,
indépendamment des autres. Quelle est alors la probabilité de marquer au moins 10 buts
avec ces 51 occasions ?

== La loi de Poisson-binomiale

Le nombre de buts suit une loi dite de *Poisson-binomiale* : une somme de tirages
indépendants, chacun avec sa propre probabilité. On la calcule exactement, tir par tir.
Avant le premier tir, on a 0 but avec probabilité 1. Chaque nouveau tir, de probabilité
$p$, envoie une part $p$ de chaque cas vers « un but de plus » et laisse le reste en
place :
$ P_n (k) = P_(n-1) (k) dot (1 - p_n) + P_(n-1) (k - 1) dot p_n $

Après 51 tirs, on a la probabilité exacte de chaque total, de 0 à 51 buts. La
*probabilité de hasard* d'un joueur est celle d'un écart au moins aussi grand que le sien,
dans le même sens : au moins autant de buts s'il sur-performe, au plus autant s'il
sous-performe. Pour Mbappé, elle vaut *5,7 %*. C'est rare, mais pas extraordinaire.

== Le piège des comparaisons multiples

On a choisi Mbappé justement parce qu'il ressortait. Or, parmi des centaines de joueurs,
quelques-uns dépasseront toujours leur xG de beaucoup, même si tous n'avaient que de la
chance. Chercher l'extrême dans une foule garantit d'en trouver un.

La bonne question n'est donc pas « tel joueur est-il sous 5 % ? » mais « y a-t-il plus de
joueurs sous 5 % que le hasard n'en fabrique tout seul ? ». Ce nombre attendu se calcule
exactement : pour chaque joueur, on additionne la probabilité, sous l'hypothèse du hasard,
de tous les totaux de buts qui le feraient passer sous 5 %.

Parmi les 137 joueurs qui ont au moins 10 tirs, *7* sont sous 5 %. Le hasard seul en
produirait environ *4*. L'excès est faible : sur ces tournois, on ne voit presque pas de
trace mesurable d'un talent de finition. Les études sur des saisons entières montrent
qu'il existe chez quelques finisseurs d'élite, mais qu'il est plus rare et plus faible que
ce que suggèrent des échantillons courts comme celui-ci.

#retenir[Un écart buts − xG doit toujours être comparé à ce que le hasard produit. Et
quand on examine beaucoup de joueurs, on compare le nombre d'écarts « surprenants » au
nombre qu'on attendrait par pure chance.]

// ---------------------------------------------------------------- 10

= Expliquer une prédiction

Pourquoi ce tir vaut-il 0,13 xG et celui-là 0,98 ? Les *valeurs SHAP* répondent en
répartissant la prédiction entre les features. On part du « tir moyen », puis chaque
feature ajoute ou retire quelque chose selon que sa valeur est au-dessus ou en dessous de
la moyenne.

Pour une régression logistique, cette répartition a une formule exacte. La contribution
de la feature $j$ à un tir est son coefficient multiplié par l'écart entre sa valeur pour
ce tir et sa valeur moyenne :
$ phi_j = c_j (x_j - macron(x)_j) $

Les contributions s'additionnent sur l'échelle des log-cotes (le score $s$ de la
section 5.1) : le score du tir moyen plus la somme des $phi_j$ redonne exactement le
score du tir, donc son xG. Un test automatique le vérifie. Pour une feature en one-hot,
on additionne les contributions de toutes ses colonnes : on lit « partie du corps :
tête » et non trois lignes séparées.

#figure(image("../figures/explication_tir.png", width: 72%),
  caption: [Une tête de Mbappé contre l'Australie (Coupe du monde 2022, 4-1), marquée à la 68#super[e] minute.
  Partant du tir moyen (0,06), chaque feature ajoute (bleu) ou retire (rouge) de
  la probabilité de but, jusqu'à l'xG du tir (0,13).]) <fig-explication>

Sur la @fig-explication, l'angle et la distance font plus que doubler l'xG du tir moyen ;
le duel aérien, le défenseur tout proche et la tête le font redescendre. Le modèle juge
cette occasion à 0,13 : environ une chance sur huit.

Une subtilité : les contributions s'additionnent en log-cotes, pas en probabilité. Le
graphique les convertit en probabilité étape par étape, dans l'ordre affiché, de la plus
forte à la plus faible. Dans un autre ordre, les barres n'auraient pas exactement la même
longueur, mais le point d'arrivée serait le même.

// ---------------------------------------------------------------- 11

= L'application

Une application Streamlit permet d'explorer le modèle sans écrire de code. Elle a trois
onglets :

- *Joueurs* : buts, xG, écart et probabilité de hasard de chaque joueur, avec le nombre de
  joueurs qu'on attendrait sous 5 % par pure chance ;
- *Tirs d'un joueur* : sa carte de tirs ; un clic sur un tir affiche son explication
  (section 10) ;
- *Le modèle* : métriques et calibration.

Tous les xG affichés sont des prédictions hors-pli : aucun tir n'est jugé par un modèle
qui a vu son match.

// ---------------------------------------------------------------- 12

= Limites et pistes

*Limites.*
- 5 583 tirs de quatre tournois, uniquement des matchs de sélections nationales : c'est
  peu pour un modèle d'xG, et le style de jeu peut différer des championnats de clubs.
- Pas de hauteur du ballon à la frappe, ni de description fine de la position de tous les
  défenseurs.
- Les annotations « but vide » et « face-à-face » sont des jugements d'analystes.
- Les penalties sont exclus du modèle et des bilans de joueurs.

*Pistes.*
- Ajouter des saisons de championnat, disponibles en partie dans les données ouvertes.
- Décrire le freeze frame plus finement : angle de tir réellement libre entre les
  défenseurs, position du gardien par rapport à la trajectoire.
- Modéliser le talent de finition directement, en ajoutant un effet par joueur au modèle
  et en mesurant s'il améliore les prédictions sur des matchs jamais vus.

// ---------------------------------------------------------------- Glossaire

#heading(numbering: none)[Glossaire]

#set par(justify: false)
#show table: set block(breakable: true)
#table(
  columns: (auto, 1fr), align: (left, left), inset: (x: 6pt, y: 5pt),
  table.hline(stroke: 0.8pt + rule),
  [*AUC (ROC)*], [Probabilité qu'un but ait un xG plus élevé qu'un tir raté pris au hasard. Juge l'ordre, pas les probabilités.],
  [*Baseline*], [Modèle le plus simple possible, qui sert de point de comparaison.],
  [*Bootstrap*], [Rééchantillonnage avec remise pour estimer l'incertitude d'une mesure.],
  [*Brier (score de)*], [Moyenne des carrés des écarts entre probabilité prédite et résultat.],
  [*Calibration*], [Accord entre les probabilités annoncées et les fréquences observées.],
  [*Coefficient*], [Poids d'une feature dans la régression logistique.],
  [*Comparaisons multiples*], [Plus on teste de cas, plus on trouve d'écarts « surprenants » par hasard.],
  [*Facteur de confusion*], [Variable qui semble avoir un effet parce qu'elle suit une autre variable.],
  [*Feature*], [Nombre ou catégorie qui décrit un tir et que le modèle utilise.],
  [*Fuite d'information*], [Information sur la réponse qui se glisse dans les données d'entraînement ou les features.],
  [*Gradient boosting*], [Somme d'arbres de décision, chacun corrigeant les erreurs des précédents.],
  [*Hors-pli*], [Prédiction faite par un modèle qui n'a pas vu la donnée prédite, ni son match.],
  [*Hyperparamètre*], [Réglage d'un modèle choisi avant l'entraînement (nombre d'arbres, $C$…).],
  [*Imputation*], [Remplacement d'une valeur manquante, ici par la médiane.],
  [*Intervalle de confiance*], [Fourchette de valeurs plausibles pour une mesure, compte tenu du hasard.],
  [*Log loss*], [Moyenne de $-log$ de la probabilité donnée au résultat réel. Punit la confiance mal placée.],
  [*Log-cote*], [Score $s$ de la logistique, logarithme de $p slash (1-p)$.],
  [*One-hot*], [Une colonne 0/1 par catégorie d'une variable.],
  [*Poisson-binomiale (loi)*], [Loi du nombre de succès parmi des tirages indépendants de probabilités différentes.],
  [*Régularisation*], [Pénalité qui retient les coefficients près de 0 pour éviter le surapprentissage.],
  [*SHAP (valeurs)*], [Répartition d'une prédiction entre les features, à partir d'un point de départ moyen.],
  [*Standardisation*], [Ramener une variable à une moyenne de 0 et un écart-type de 1.],
  [*Surapprentissage*], [Un modèle qui apprend le bruit de ses données d'entraînement et généralise mal.],
  [*Validation croisée*], [Estimer la performance en tournant sur plusieurs découpages des données d'entraînement.],
  [*xG*], [_Expected goals_ : probabilité qu'un tir finisse au fond.],
  table.hline(stroke: 0.8pt + rule),
)

#v(1em)
#text(size: 9.5pt, fill: ink-secondary)[Données : StatsBomb Open Data,
#link("https://github.com/statsbomb/open-data")[github.com/statsbomb/open-data]. Tous les
chiffres de ce rapport sont reproductibles à partir du dépôt.]
