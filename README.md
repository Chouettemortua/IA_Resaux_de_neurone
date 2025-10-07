# Projet IA Sommeil

Ce projet contient des modèles d’intelligence artificielle pour prédire la qualité du sommeil et détecter des troubles du sommeil à partir de données utilisateur.

---

## Organisation du repo

- `Code/Saves/` : modèles entraînés :fichiers pickle
- `Code/Saves_Curves/` : courbe d'évaluation de performance des différent modèle
- `Code/Saves_user/` : emplacement de sauvegarde de csv de test
- `Code/Data/` : datasets utilisés pour l’entraînement
- `app.py` : code pour [l'app en ligne](https://huggingface.co/spaces/Chouettemortua/IA_Sleep)
- `Code/Core/models/AI_Model.py` : code pour la création des model IA et fonction utilitaire
- `Code/Core/preprocessing/preprocessing.py` : code qui s'occupe de traiter les données brute
- `Code/Core/training/AI_training.py` : lance les entrainement/création des IA
- `Code/Core/training/training_utils.py` : Quelque code utilitaire pour la gestion de l'entrainement des IA
- `Code/Core/utils/utils.py` : des code utilitaire plus globaux
- `Code/Gravewards` : de fichier de code qui ne sont plus utiliser dans le projet mais on un intéret pour montrer la progression ou certaine idée
- `Code/app_desk.py` : interface en PYQT6 pour une app desktop orienté utilisateurs
- `Code/run.py` : interface en PYQT6 pour une app desktop administrateur donc avec tout pour les test et les entrainement d'IA
- `Code/test.py` : code temporaire
- `presentation/` : Fichier pour la présentation du projet
- `Analyse/` : png d'analyse du dataset pour l'entrainement
- `Journal/` : Journal d'avancement et autre fichier text de progression du projet

---

## Dataset

Le dataset utilisé pour l’entraînement est disponible dans le dossier [`TIPE/datasets/Sleep_health_and_lifestyle_dataset.csv`](TIPE/datasets/Sleep_health_and_lifestyle_dataset.csv).

Ce dataset provient de [Kaggle - Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset/data) et est distribué sous la licence **CC0: Public Domain**.
Cela signifie que ce dataset est libre de droits et peut être utilisé, modifié, et partagé sans restriction.

---

## Documents légaux

- [Conditions Générales d’Utilisation (CGU)](CGU.md)
- [Politique de Confidentialité](PRIVACY_PRIVACY.md)
- [Licence d’Utilisation](LICENSE.md)

Ces documents contiennent les informations importantes concernant l'utilisation du logiciel, la gestion des données personnelles, ainsi que les droits et obligations de l’utilisateur.

---

## Contact

# Pour toute question ou demande, contactez [[roche.ewann@gmail.com](mailto:roche.ewann@gmail.com)]
