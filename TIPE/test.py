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