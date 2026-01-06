__init__ = "Gradio app module"

import gradio as gr
import pandas as pd
import numpy as np
import tempfile
import os

from Code.Core.training.training_utils import model_charge
from Code.Core.preprocessing.preprocessing import preprocecing_user

class GradioApp:
    def __init__(self):
        self.model_T = model_charge('Code/Saves/save_sleep_trouble.pkl')
        self.model_Q = model_charge('Code/Saves/save_sleep_quality.pkl')

        self.columns = [
            "Gender", "Age", "Occupation", "Sleep Duration", "Physical Activity Level",
            "Stress Level", "BMI Category", "Blood Pressure", "Heart Rate", "Daily Steps"
        ]

        self.df = pd.DataFrame(columns=self.columns)

        self.label_trouble = {
            0:"Normal (pas de trouble)",
            1:"Apnée du sommeil",
            2:"Insomnie",
        }

    def add_entry(self, df_data, *args):
        keys = ["Gender", "Age", "Occupation", "Sleep Duration", "Physical Activity Level", "Stress Level", "BMI Category", "Blood Pressure", "Heart Rate", "Daily Steps"]
        kwargs = dict(zip(keys, args))
        df = pd.DataFrame(df_data, columns=self.columns)
        df_new = pd.concat([df, pd.DataFrame([kwargs])], ignore_index=True)
        return df_new
    
    def analyze_data(self, df_data):
        df = pd.DataFrame(df_data, columns=self.columns)
        
        if df.empty:
            return "Aucune donnée à analyser."

        # Prendre les 5 dernières entrées ou toutes si il y en a moins de 5
        n = min(5, len(df))
        recent_entries = df.tail(n)

        try:
            # Prétraitement de toutes les entrées récentes
            df_processed = preprocecing_user(recent_entries)
            
            # Prédictions et calcul des moyennes
            pred_qualities = self.model_Q.predict(df_processed)
            mean_quality = np.mean(pred_qualities)
            
            pred_troubles = self.model_T.predict(df_processed)
            mean_trouble = np.mean(pred_troubles)
            
            # Détermination de la classe de trouble dominante en arrondissant la moyenne
            classe_moyenne_trouble = int(round(mean_trouble))
            classe_moyenne_quality = int(round(mean_quality * 10))
            label_trouble = self.label_trouble.get(classe_moyenne_trouble, "Inconnu")

            return (
                f"Analyse basée sur les {n} dernières entrées :\n"
                f"Score moyen qualité de sommeil : {classe_moyenne_quality}%\n"
                f"Trouble du sommeil dominant : {label_trouble}"
            )

        except Exception as e:
            gr.Error(f"Échec de l'analyse : {str(e)}")
            return f"Échec de l'analyse : {str(e)}"
        
    def load_csv(self, file):
        if file is None:
            return pd.DataFrame(columns=self.columns)
        
        try:
            # Gradio passe un objet tempfile, on accède au chemin via .name
            df_loaded = pd.read_csv(file.name)
            
            # On s'assure que les colonnes matchent, sinon on réordonne/complète
            for col in self.columns:
                if col not in df_loaded.columns:
                    df_loaded[col] = np.nan
            
            return df_loaded[self.columns]
        except Exception as e:
            raise gr.Error(f"Erreur de lecture : {e}")

    def save_csv(self, df_data):
        # 1. Vérification : si df_data est vide ou None
        if df_data is None or (isinstance(df_data, pd.DataFrame) and df_data.empty):
            # On peut renvoyer None ou un fichier vide pour éviter le crash
            return None 

        # 2. Conversion explicite en DataFrame au cas où Gradio envoie une liste
        df = pd.DataFrame(df_data, columns=self.columns)
        
        # 3. Création du fichier temporaire
        file_path = os.path.join(tempfile.gettempdir(), "sleep_data_export.csv")
        df.to_csv(file_path, index=False)
        
        # 4. On retourne UNIQUEMENT le string du chemin
        return file_path

    def clear_data(self):
        return pd.DataFrame(columns=self.columns)

    def launch(self):
        with gr.Blocks() as demo:
            gr.Markdown("# Application d'Analyse du Sommeil")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Actions sur les Fichiers")
                    with gr.Group():
                        load_file_button = gr.UploadButton("Charger un CSV", file_types=[".csv"])
                        save_file_button = gr.DownloadButton("Sauvegarder en CSV")
                        clear_button = gr.Button("Vider la table")
                        
                
                with gr.Column(scale=1):
                    gr.Markdown("### Analyse")
                    analyze_button = gr.Button("Analyser")
                    # Ajout de lines=3 pour augmenter la hauteur de la zone de texte
                    analysis_output = gr.Textbox(label="Résultats de l'Analyse", lines=3)
            
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("## Tableau de Données")
                    table = gr.Dataframe(
                        headers=self.columns,
                        row_count=10,
                        col_count=(len(self.columns), "fixed"),
                        type="pandas",
                        label="Données",
                        interactive=True,
                        value=pd.DataFrame(columns=self.columns)
                    )
                with gr.Column(scale=1):
                    gr.Markdown("## Formulaire d'Entrée")
                    with gr.Group():
                        # Dictionnaire des inputs pour le bouton d'ajout
                        input_components_add = {
                            "Gender": gr.Dropdown(choices=["Male", "Female"], label="Genre", value="Male"),
                            "Age": gr.Number(label="Âge", value=30, minimum=0, maximum=120),
                            "Occupation": gr.Dropdown(choices=['working', 'unemployed', 'student', 'retired', 'other'], label="Occupation", value="working"),
                            "Sleep Duration": gr.Number(label="Durée du Sommeil (heures)", value=7.0, minimum=0.0, maximum=24.0),
                            "Physical Activity Level": gr.Number(label="Activité Physique (minutes par jour)", value=30, minimum=0),
                            "Stress Level": gr.Number(label="Niveau de Stress", value=5, minimum=1, maximum=10),
                            "BMI Category": gr.Dropdown(choices=["Normal", "Overweight", "Underweight", "Obese"], label="IMC", value="Normal"),
                            "Blood Pressure": gr.Textbox(label="Pression Artérielle (Ex: 120/80)", value="120/80"),
                            "Heart Rate": gr.Number(label="Fréquence Cardiaque (bpm)", value=70, minimum=30, maximum=200),
                            "Daily Steps": gr.Number(label="Pas Quotidiens", value=3000, minimum=0, step=1000)
                        }
                    add_button = gr.Button("Ajouter à la base")
            

            # Définition des interactions
            add_button.click(
                fn=self.add_entry,
                inputs=[table, *input_components_add.values()],
                outputs=table
            )
            
            analyze_button.click(
                fn=self.analyze_data,
                inputs=table,
                outputs=analysis_output
            )

            save_file_button.click(
                fn=self.save_csv,
                inputs=[table],      
                outputs=save_file_button
            )
            
            load_file_button.upload(
                fn=self.load_csv,
                inputs=[load_file_button],
                outputs=[table]
            )
            
            clear_button.click(
                fn=self.clear_data,
                inputs=None,
                outputs=table
            )
            
        demo.launch()


app = GradioApp()
app.launch()