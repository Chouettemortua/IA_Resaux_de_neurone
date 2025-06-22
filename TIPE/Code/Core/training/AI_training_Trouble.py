

import sys
import os
import numpy as np
sys.path.append(os.path.abspath('/home/chouettemortua/IA_Resaux_de_neurone/TIPE/Code/Core/preprocessing'))
from preprocessing import preprocecing
from training_utils import model_charge, load, model_init, model_train, affichage_perf
sys.path.append(os.path.abspath('/home/chouettemortua/IA_Resaux_de_neurone/TIPE/Code/Core/utils'))
from utils import courbe_perf

def main_sleep_trouble(boul_c, bool_t, path_n, path_c):
    """ Main function for Resaux on the sleep trouble dataset """
    
    # Load the dataset
    data = load('TIPE/Code/Data/Sleep_health_and_lifestyle_dataset.csv')
    df = data.copy()

    # Preprocessing
    # Uncomment the following line to see the dataset before preprocessing
    # analyse_pre_process(df)
    
    X_train, y_train, X_test, y_test = preprocecing(df, ['Sleep Disorder'], y_normalisation=False)

    assert not np.any(np.isin(X_train.index, X_test.index))

    # Uncomment the following line to see the dataset after preprocessing
    # analyse_post_process(X_train, y_train, X_test, y_test)
    
    # Train the model

    # architecture = [1,30,75,500,1000,500,100,75,30,1] atteint les 0.8 
    # architecture = [1,2000,1500,1000,500,400,100,75,30,1] atteint les 0.85 mais pas stable et long -> set trop petit pour le nombre de neurones
    # architecture = [1, 64, 32, 16, 1] se stabilise à 0.5 sous apprentissage
    
    if boul_c:
        sleep = model_init(path_n, X_train, y_train, X_test, y_test, [256, 128, 64, 32, 16, 4], path_n, treshold_val=None)
    else: 
        sleep = model_charge(path_n)

    if bool_t:
        sleep = model_train(X_train, y_train, X_test, y_test, sleep, path_n, iteration=1000, precision=1e-2)

    # Ami évaluation affichage
    ami = [0,0.3,0.6,0.47,0.8,0.11,4,0,0.88,0.89,0.5]
    print(f"ami: {(sleep.predict(np.array(ami).reshape(1,-1)))}")

    # affichage des performances

    affichage_perf(X_train, y_train, X_test, y_test, sleep)

    # courbe_perf(sleep)
    if boul_c or bool_t:     
        courbe_perf(sleep,path_c)

# Lancement automatique des fonctions principales
if __name__ == "__main__":
    # Main function launcher with arguments
    main_sleep_trouble(False, False, "TIPE/Code/Saves/save_sleep_trouble.pkl", "TIPE/Code/Saves_Curves/curve_sleep_trouble.png")