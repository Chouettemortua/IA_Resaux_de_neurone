import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des données
df = pd.read_csv('TIPE/datasets/Sleep_health_and_lifestyle_dataset.csv')

# Création de l'histogramme
plt.figure(figsize=(10, 6))
sns.histplot(df['Quality of Sleep'], bins=10, kde=True, color='skyblue')
plt.title('Distribution de la qualité du sommeil')
plt.xlabel('Note de qualité du sommeil')
plt.ylabel('Fréquence')
plt.grid(True)
plt.show()