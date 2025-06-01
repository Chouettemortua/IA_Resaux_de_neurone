import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import AI_Model
matplotlib.use('QtAgg')

# Chargement des données
df = pd.read_csv('TIPE/datasets/Sleep_health_and_lifestyle_dataset.csv')
'''
df = df.drop(columns=['Quality of Sleep'], axis=1)
df = df.drop(columns=['Sleep Disorder'], axis=1)
df = AI_Model.preprocecing_user(df)
'''

# Création de l'histogramme
plt.figure(figsize=(10, 6))
sns.histplot(df['Gender'], bins=10, kde=True, color='skyblue')
plt.title('Distribution of Gender')
plt.xlabel('Genre')
plt.ylabel('Fréquence')
plt.grid(True)
plt.show()


def recommander_améliorations(model, input_row, features, non_modifiables):
    base_pred = model.predict(input_row.reshape(1, -1))[0]
    impacts = {}

    for i, feature in enumerate(features):
        if feature in non_modifiables:
            continue

        val = input_row[i]
        # Variations tests : +5% et -5%
        for delta in [-0.05 * val, 0.05 * val]:
            modified = input_row.copy()
            modified[i] = val + delta
            new_pred = model.predict(modified.reshape(1, -1))[0]
            impact = new_pred - base_pred
            # On ne garde que les impacts positifs (amélioration)
            if impact > 0:
                impacts[feature] = max(impacts.get(feature, 0), impact)

    # Trier par impact décroissant
    impacts_tries = sorted(impacts.items(), key=lambda x: x[1], reverse=True)

    # Construire message ou interface avec top suggestions
    recommandations = []
    for feature, impact in impacts_tries:
        recommandations.append(f"Augmenter {feature} pourrait améliorer votre score de {impact:.4f}")

    return recommandations
