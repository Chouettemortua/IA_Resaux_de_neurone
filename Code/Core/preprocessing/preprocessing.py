__init__ = "preprocessing"

import pandas as pd
from sklearn.model_selection import train_test_split

# Pretraitement des données

def preprocecing(df, on, y_normalisation=True, debug=False):
    """ Prétraite les données pour l'entrainement et le test 
        args:
            df: dataframe pandas
            on: liste des colonnes à prédire
            y_normalisation: booléen, si True normalise y entre 0 et 1
            debug: booléen, si True affiche des informations de debug
        returns:
            X_train, y_train, X_test, y_test: données d'entrainement et de test
    """

    def encodage(df):
        """ Encode les variables catégorielles """

        # Dictionnaires de mapping adaptés au dataset initial (entrainement)
        code_bmi = {'Normal':0,'Normal Weight': 0, 'Overweight': 2, 'Underweight': 3, 'Obesity': 4}
        code_gender = {'Male': 0, 'Female': 1}
        code_occupation = {'Software Engineer': 0, 'Doctor': 0, 'Sales Representative': 0, 'Nurse': 0, 'Teacher': 0,
                        'Scientist': 0, 'Engineer': 0, 'Lawyer': 0, 'Accountant': 0, 'Salesperson': 0, 'Manager': 0}
        code_sleep_disorder = {'Normal': 0, 'Sleep Apnea': 1, 'Insomnia': 2}
        
        # Nettoyage et conversion de Blood Pressure
        df['Blood Pressure'] = df['Blood Pressure'].str.split('/').str[0].astype(int)
        df['Sleep Disorder'] = df['Sleep Disorder'].apply(lambda x: x if x in ['Sleep Apnea', 'Insomnia'] else 'Normal')
        
        # Mapper les colonnes catégorielles
        df['BMI Category'] = df['BMI Category'].map(code_bmi).fillna(0).astype(int)
        df['Gender'] = df['Gender'].map(code_gender).fillna(0).astype(int)
        df['Occupation'] = df['Occupation'].map(code_occupation).fillna(0).astype(int)
        df['Sleep Disorder'] = df['Sleep Disorder'].map(code_sleep_disorder).fillna(0).astype(int)

        return df

    def imputation(df):
        """ Impute les valeurs manquantes et supprime les colonnes inutiles """
        df = df.drop(columns=['Person ID'])
        return df.fillna(df.mean())
    
    def split_data(df):
        """ Sépare le dataset en train et test """

        trainset, testset = train_test_split(df, test_size=0.2, random_state=42)
        return trainset, testset
    
    def normalisation(df):
        """ Normalise les données entre 0 et 1 """

        # Valeurs max pour chaque feature encoder manuellement pour correspondre a des valeurs physiologique réalistes (à ajuster si besoin)
        # pour assurer une meilleure généralisation du modèle
        max_values = {'Gender': 2, 'Age': 130, 'Occupation': 5, 'Sleep Duration': 24, 
                      'Physical Activity Level': 360, 'Stress Level': 10, 'BMI Category': 4, 
                      'Blood Pressure': 200, 'Heart Rate': 200, 'Daily Steps': 50000, 'Quality of Sleep': 10}

        # Normalisation des colonnes
        df['Gender'] = df['Gender'].div(max_values['Gender'])
        df['Age'] = df['Age'].div(max_values['Age'])
        df['Occupation'] = df['Occupation'].div(max_values['Occupation'])
        df['Sleep Duration'] = df['Sleep Duration'].div(max_values['Sleep Duration'])
        df['Physical Activity Level'] = df['Physical Activity Level'].div(max_values['Physical Activity Level'])
        df['Stress Level'] = df['Stress Level'].div(max_values['Stress Level'])
        df['BMI Category'] = df['BMI Category'].div(max_values['BMI Category'])
        df['Blood Pressure'] = df['Blood Pressure'].div(max_values['Blood Pressure'])
        df['Heart Rate'] = df['Heart Rate'].div(max_values['Heart Rate'])
        df['Daily Steps'] = df['Daily Steps'].div(max_values['Daily Steps'])


        return df
    
    def normalisation_y(y):
        """ Normalise y entre 0 et 1 """
        col_name = on[0]
        max_values = {'Quality of Sleep': 10, 'Sleep Disorder': 2}
        if col_name in max_values:
            return y / max_values[col_name]
        return y
    
    def intern(df):
        """ Prétraite les données et sépare X et y """

        # Encodage des variables catégorielles et imputation des valeurs manquantes
        df= encodage(df)
        df = imputation(df)

        # Séparation des features et de la target
        X = df.drop(columns= on, axis=1)
        y = df[on[0]].values.reshape(-1, 1)

        # Normalisation
        X = normalisation(X)
        if y_normalisation:
            # Normalisation de y si demandé
            y = normalisation_y(y) 
        return X, y

    # Séparation train/test
    trainset, testset = split_data(df)
    # Prétraitement des données train et test
    X_train, y_train = intern(trainset)
    X_test, y_test = intern(testset)

    if debug:
        print("Train size:", X_train.shape, "Test size:", X_test.shape)
        print("Distribution:", pd.Series(y_train.flatten()).value_counts(normalize=True))
    

    return X_train, y_train, X_test, y_test   

def preprocecing_user(df, on=None):
    """ Prétraite les données qui proviennent de l'utilisateur
        args:
            df: dataframe pandas
            on: liste des colonnes à prédire (optionnel car les données utilisateur peuvent ne pas les contenir) 
        returns:
            df: dataframe pandas prétraitée"""

    def encodage(df):
        """ Encode les variables catégorielles """

        # Dictionnaires de mapping adaptés aux entrées utilisateurs
        code_bmi = {'Normal': 0, 'Normal Weight': 0, 'Overweight': 1, 'Underweight': 2, 'Obese': 3}
        code_gender = {'Male': 0, 'Female': 1}
        code_occupation = {'working':0, 'unemployed':0, 'student':0, 'retired':0, 'other':0} # tout a 0 car absence de données suffisantes (on a que des working dans le dataset initial), si nouvelle donnée a ajuster

        df['Blood Pressure'] = df['Blood Pressure'].astype(str).str.split('/').str[0]
        df['Blood Pressure'] = pd.to_numeric(df['Blood Pressure'], errors='coerce')


        # Mapper les colonnes catégorielles

        # test debug (line à décommenter si besoin)
        #print("BMI uniques reçus :", df['BMI Category'].unique())
        
        df['BMI Category'] = df['BMI Category'].astype(str).str.strip().map(code_bmi)
        if df['BMI Category'].isnull().any():
            raise ValueError("Valeur invalide dans 'BMI Category'. Vérifiez vos entrées.")
        df['BMI Category'] = df['BMI Category'].astype(int)

        df['Gender']  = df['Gender'].astype(str).str.strip().map(code_gender)
        if df['Gender'].isnull().any():
            raise ValueError("Valeur invalide dans 'Gender'. Vérifiez vos entrées.")
        df['Gender'] = df['Gender'].astype(int)

        df['Occupation'] = df['Occupation'].astype(str).str.strip().map(code_occupation)
        if df['Occupation'].isnull().any():
            raise ValueError("Valeur invalide dans 'Occupation'. Vérifiez vos entrées.")
        df['Occupation'] = df['Occupation'].astype(int)
        

        return df

    def imputation(df):
        """ Impute les valeurs manquantes et supprime les colonnes inutiles """
        return df.fillna(df.mean())
    
    def normalisation(df):
        """ Normalise les données entre 0 et 1 """

        max_values = {'Gender': 2, 'Age': 130, 'Occupation': 5, 'Sleep Duration': 24, 
                      'Physical Activity Level': 360, 'Stress Level': 10, 'BMI Category': 4, 
                      'Blood Pressure': 200, 'Heart Rate': 200, 'Daily Steps': 50000}
       
        df['Gender'] = df['Gender'].div(max_values['Gender'])
        df['Age'] = df['Age'].div(max_values['Age'])
        df['Occupation'] = df['Occupation'].div(max_values['Occupation'])
        df['Sleep Duration'] = df['Sleep Duration'].div(max_values['Sleep Duration'])
        df['Physical Activity Level'] = df['Physical Activity Level'].div(max_values['Physical Activity Level'])
        df['Stress Level'] = df['Stress Level'].div(max_values['Stress Level'])
        df['BMI Category'] = df['BMI Category'].div(max_values['BMI Category'])
        df['Blood Pressure'] = df['Blood Pressure'].div(max_values['Blood Pressure'])
        df['Heart Rate'] = df['Heart Rate'].div(max_values['Heart Rate'])
        df['Daily Steps'] = df['Daily Steps'].div(max_values['Daily Steps'])

        return df
    
    def intern(df):
        """ Prétraite les données """
        if on is not None:
            df.drop(columns=on, axis=1, inplace=True)
        
        df= encodage(df)
        df = imputation(df)
        df = normalisation(df) 
        return df

    return intern(df)
