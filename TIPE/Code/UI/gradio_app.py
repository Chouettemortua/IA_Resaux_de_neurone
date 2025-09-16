
__init__ = "Gradio app module"

import gradio as gr
import sys
import os
import pandas as pd

from Core.training.training_utils import model_charge
from Core.preprocessing.preprocessing import preprocecing_user

class GradioApp:
    def __init__(self):
        self.model_T = model_charge('TIPE/Code/Saves/save_sleep_trouble.pkl')
        self.model_Q = model_charge('TIPE/Code/Saves/save_sleep_quality.pkl')

    def predict(self, input_data):
        # Convertir les données d'entrée en DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Prétraiter les données d'entrée
        X_processed = preprocecing_user(input_df)
        
        # Faire la prédiction 
        prediction_T = self.model_T.predict(X_processed)
        prediction_Q = self.model_Q.predict(X_processed)
        return "Qualité du Sommeil: {:.2f}".format(prediction_Q[0]) + "\nTrouble du Sommeil: {:.2f}%".format(prediction_T[0])


    def launch(self):
        with gr.Blocks() as demo:
            gr.Markdown("# Prédiction des Troubles et Qualité du Sommeil")
            input_data = {
                "Gender": gr.Dropdown(choices=["Male", "Female"], label="Genre", value="Male"),
                "Age": gr.Number(label="Âge", value=30),
                "Occupation": gr.Dropdown(choices=["working", "unemployed", "student", "retired", "other"], label="Occupation", value="working"),
                "Sleep Duration": gr.Number(label="Durée du Sommeil (heures)", value=7),
                "Physical Activity Level": gr.Number(label="Activité Physique (minutes par jour)", value=30),
                "Stress Level": gr.Number(label="Niveau de Stress", value="5"),
                "BMI Category": gr.Dropdown(choices=["Normal", "Overweight", "Underweight", "Obese"], label="IMC", value="Normal"),
                "Blood Pressure": gr.Text(label="Pression Artérielle (Ex: 120/80)", value="120/80"),
                "Heart Rate": gr.Number(label="Fréquence Cardiaque (bpm)", value=70),
                "Daily Steps": gr.Number(label="Pas Quotidiens", value=3000)
            }
            predict_button = gr.Button("Prédire")
            output = gr.Textbox(label="Résultats de la Prédiction")
            predict_button.click(fn=self.predict, inputs=input_data, outputs=output)
        demo.launch()

def run_gradio_app():
    app = GradioApp()
    app.launch()