title: TIPE_APP_SLEEP_IA
emoji: 😴
colorFrom: red
colorTo: pink
sdk: gradio
app_port: 7860
--------------

# Projet IA Sommeil

Ce projet contient des modèles d’intelligence artificielle pour prédire la qualité du sommeil et détecter des troubles du sommeil à partir de données utilisateur.

---

## Organisation du repo

- `TIPE/Saves/` : modèles entraînés :fichiers pickle, png des courbe de performance et fichier csv des donné test faite a la main
- `TIPE/datasets/` : datasets utilisés pour l’entraînement
- `TIPE/app.py` : code de l’application Qt6
- `TIPE/AI_Model.py` : code pour la création des model IA et fonction utilitaire
- `TIPE/test.py` : code temporaire
- `TIPE/presentation/` : Fichier pour la présentation du projet
- `TIPE/Analyse/` : png d'analyse du dataset pour l'entrainement
- `TIPE/Journal/` : Journal d'avancement et autre fichier text de progression du projet

---

## Dataset

Le dataset utilisé pour l’entraînement est disponible dans le dossier [`TIPE/datasets/Sleep_health_and_lifestyle_dataset.csv`](TIPE/datasets/Sleep_health_and_lifestyle_dataset.csv).

Ce dataset provient de [Kaggle - Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset/data) et est distribué sous la licence **CC0: Public Domain**.
Cela signifie que ce dataset est libre de droits et peut être utilisé, modifié, et partagé sans restriction.

---

## Documents légaux

- [Conditions Générales d’Utilisation (CGU)](CGU.md)
- [Politique de Confidentialité](PRIVACY_POLICY.md)
- [Licence d’Utilisation](LICENSE.md)

Ces documents contiennent les informations importantes concernant l'utilisation du logiciel, la gestion des données personnelles, ainsi que les droits et obligations de l’utilisateur.

---

## Contact

# Pour toute question ou demande, contactez [[roche.ewann@gmail.com](mailto:roche.ewann@gmail.com)]
