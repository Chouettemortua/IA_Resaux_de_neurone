
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from PyQt6.QtCore import QObject, pyqtSignal
import shap

from ..preprocessing.preprocessing import preprocecing
from .training_utils import model_charge, load, model_init, model_train, affichage_perf, val_evolution
from ..utils.utils import courbe_perf

class TrainingWorker(QObject):
    """
    Cette classe contient la logique d'entraînement et émet le signal de progression.
    """
    # Signal émis pour indiquer la progression de l'entraînement
    progress_updated = pyqtSignal(int)
    # Signal émis lorsque l'entraînement est terminé
    finished = pyqtSignal()

    def __init__(self, model_type, bool_c, bool_t, path_n, path_c, nb_iter, verbose):
        """
        Initialise le worker avec les paramètres nécessaires.
        """
        super().__init__()
        self.model_type = model_type
        self.bool_c = bool_c
        self.bool_t = bool_t
        self.path_n = path_n
        self.path_c = path_c
        self.nb_iter = nb_iter
        self.verbose = verbose

    def run(self):
        """
        La méthode qui lance l'entraînement.
        """
        
        # Load the dataset
        data = load('TIPE/Code/Data/Sleep_health_and_lifestyle_dataset.csv')
        df = data.copy()
        # Preprocessing
        # Uncomment the following line to see the dataset before preprocessing
        # analyse_pre_process(df)
            
        if self.model_type == "T" :
            X_train, y_train, X_test, y_test = preprocecing(df, ['Sleep Disorder', 'Quality of Sleep'], y_normalisation=False)
            self.learning_rate = 1e-3
        elif self.model_type == "Q":
            X_train, y_train, X_test, y_test = preprocecing(df, ['Quality of Sleep', 'Sleep Disorder'], y_normalisation=False)
            self.learning_rate = 1e-2
            ''' was used when quality of sleep was treat as a continuous variable
            # Transformer y_train / y_test avec QuantileTransformer
            qt = QuantileTransformer(output_distribution='normal', random_state=42, n_quantiles= 299)
            y_train = qt.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_test = qt.transform(y_test.reshape(-1, 1)).flatten()
            '''

        assert not np.any(np.isin(X_train.index, X_test.index))

        # Uncomment the following line to see the dataset after preprocessing
        # analyse_post_process(X_train, y_train, X_test, y_test)
        
        # Train the model

        # Quelques architectures testées :
        # architecture = [1,30,75,500,1000,500,100,75,30,1] atteint les 0.8 
        # architecture = [1,2000,1500,1000,500,400,100,75,30,1] atteint les 0.85 mais pas stable et long -> set trop petit pour le nombre de neurones
        # architecture = [1, 64, 32, 16, 1] se stabilise à 0.5 sous apprentissage
        
        if self.bool_c:
            if self.model_type == "T":
                self.model = model_init(self.path_n, X_train, y_train, X_test, y_test, [1024, 512, 256, 128, 64, 32, 16, 3], treshold_val=None)
            elif self.model_type == "Q":
                self.model = model_init(self.path_n, X_train, y_train, X_test, y_test, [128,64,32,16,10], treshold_val=None)
        else: 
            self.model = model_charge(self.path_n)

        # Connecte le signal du modèle à un slot temporaire pour émettre la progression
        self.model.progress_updated.connect(self.progress_updated)

        if self.bool_t:
            self.model = model_train(X_train, y_train, X_test, y_test, self.model, self.path_n, self.nb_iter, learning_rate=self.learning_rate)

        # affichage des performances
        if self.model_type == "T":
            affichage_perf(X_train, y_train, X_test, y_test, self.model, self.path_c)
        elif self.model_type == "Q":
            affichage_perf(X_train, y_train, X_test, y_test, self.model, self.path_c) 
            # evolution des variables modifiables pour améliorer la prédiction
            features = ["Gender", "Age", "Occupation", "Sleep Duration",
                        "Physical Activity Level", "Stress Level", "BMI Category",
                        "Blood Pressure", "Heart Rate", "Daily Steps"]
            
            non_modifiables = ["Quality of Sleep", "Sleep Disorder", 'Age', 'Occupation', 'Gender', 'Heart Rate', 'Blood Pressure']

            modifiable_indices = [i for i, f in enumerate(features) if f not in non_modifiables]
            modifiable_features = [features[i] for i in modifiable_indices]

            # Ami sert de test
            ami = [0,0.3,0.2,0.47,0.11,4,0,0.88,0.89,0.5]
            ami_in = np.array(ami).reshape(1,-1)

            if self.verbose:
                print("")
                print("Evolution des variables modifiables pour améliorer la prédiction :")
                val_evolution(self.model, ami_in, modifiable_features, modifiable_indices, features, nb_iter=30)
                print(ami_in.shape)

                # explainer
                explainer = shap.KernelExplainer(self.model.predict, X_train)

                # Calculer les valeurs SHAP
                shap_values = explainer.shap_values(ami_in, nsamples=100)

                # Visualiser
                shap.initjs()
                shap.force_plot(explainer.expected_value, shap_values, ami_in)
                shap.summary_plot(shap_values, features=features, feature_names=features, show=False)
                #plt.savefig('TIPE/Code/Saves_Curve/values.png', bbox_inches='tight')
                #plt.close()

        # courbe_perf(sleep)
        if self.bool_c or self.bool_t:     
            courbe_perf(self.model,self.path_c)

        self.finished.emit()
        