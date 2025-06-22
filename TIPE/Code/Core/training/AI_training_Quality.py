

from sklearn.preprocessing import QuantileTransformer
import shap

import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

sys.path.append(os.path.abspath('/home/chouettemortua/IA_Resaux_de_neurone/TIPE/Code/Core/preprocessing'))
from preprocessing import preprocecing
from training_utils import model_charge, load, model_init, model_train, affichage_perf, val_evolution
sys.path.append(os.path.abspath('/home/chouettemortua/IA_Resaux_de_neurone/TIPE/Code/Core/utils'))
from utils import courbe_perf

def main_quality_of_sleep(bool_c, bool_t, path_n, path_c, verbose=False):
    """ Main function for Resaux on the quality of sleep dataset """
    
    # Load the dataset

    data = load('TIPE/Code/Data/Sleep_health_and_lifestyle_dataset.csv')
    df = data.copy()

    # Define ami

    ami = [0,0.3,0.2,0.47,0.11,4,0,0.88,0.89,0.5]

    #Preprocessing

    # Uncomment the following line to see the dataset before preprocessing
    # analyse_pre_process(df)
    
    X_train, y_train, X_test, y_test = preprocecing(df, ['Quality of Sleep', 'Sleep Disorder'], y_normalisation=True)

    assert not np.any(np.isin(X_train.index, X_test.index))

    # Uncomment the following line to see the dataset after preprocessing
    # analyse_post_process(X_train, y_train, X_test, y_test)

    # Transformer y_train / y_test avec QuantileTransformer
    qt = QuantileTransformer(output_distribution='normal', random_state=42)
    y_train = qt.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = qt.transform(y_test.reshape(-1, 1)).flatten()
    
    # Train the model
    if bool_c:
        sleep = model_init(path_n, X_train, y_train, X_test, y_test, [128,64,32,16,1], path_n, treshold_val=0.5, qt=qt)
    else: 
        sleep = model_charge(path_n)
    if bool_t:
        sleep = model_train(X_train, y_train, X_test, y_test, sleep, path_n)

    # Ami évaluation affichage

    ami_in = np.array(ami).reshape(1,-1)
    ami_pred = sleep.predict(ami_in)[-1].flatten()
    print(f"mons amis: {ami_pred}") 

    # affichage des performances

    affichage_perf(X_train, y_train, X_test, y_test, sleep, qt)    

    # evolution des variables modifiables pour améliorer la prédiction
    features = ["Gender", "Age", "Occupation", "Sleep Duration",
                "Physical Activity Level", "Stress Level", "BMI Category",
                "Blood Pressure", "Heart Rate", "Daily Steps"]
    
    non_modifiables = ["Quality of Sleep", "Sleep Disorder", 'Age', 'Occupation', 'Gender', 'Heart Rate', 'Blood Pressure']

    modifiable_indices = [i for i, f in enumerate(features) if f not in non_modifiables]
    modifiable_features = [features[i] for i in modifiable_indices]

    if verbose:
        print("")
        print("Evolution des variables modifiables pour améliorer la prédiction :")
        val_evolution(sleep, ami_in, modifiable_features, modifiable_indices, features, nb_iter=30)
        print(ami_in.shape)

        # explainer
        explainer = shap.KernelExplainer(sleep.predict, X_train)

        # Calculer les valeurs SHAP
        shap_values = explainer.shap_values(ami_in, nsamples=100)

        # Visualiser
        shap.initjs()
        shap.force_plot(explainer.expected_value, shap_values, ami_in)
        shap.summary_plot(shap_values, features=features, feature_names=features, show=False)
        plt.savefig('TIPE/Code/Saves_Curve/values', bbox_inches='tight')
        plt.close()


    #courbe_perf(sleep)
    if bool_c or bool_t:     
        courbe_perf(sleep, path_c)

# Lancement automatique des fonctions principales
if __name__ == "__main__":
    # Main function launcher with arguments
    main_quality_of_sleep(False, False, "TIPE/Code/Saves/save_sleep_quality.pkl", "TIPE/Code/Saves_Curves/curve_sleep_quality.png")