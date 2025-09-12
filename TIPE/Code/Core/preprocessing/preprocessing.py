__init__ = "preprocessing"

import pandas as pd
from sklearn.model_selection import train_test_split

# Pretraitement des données

def preprocecing(df, on, y_normalisation=True):
    """ Prétraite les données """

    def encodage(df):
        """ Encode les variables catégorielles """

        code_bmi = {'Normal':0,'Normal Weight': 0, 'Overweight': 2, 'Underweight': 3, 'Obesity': 4}
        code_gender = {'Male': 0, 'Female': 1}
        code_occupation = {'Software Engineer': 0, 'Doctor': 0, 'Sales Representative': 0, 'Nurse': 0, 'Teacher': 0,
                        'Scientist': 0, 'Engineer': 0, 'Lawyer': 0, 'Accountant': 0, 'Salesperson': 0, 'Manager': 0}
        code_sleep_disorder = {'Normal': 0, 'Sleep Apnea': 1, 'Insomnia': 2}
        

        df['Blood Pressure'] = df['Blood Pressure'].str.split('/').str[0].astype(int)
        df['Sleep Disorder'] = df['Sleep Disorder'].apply(lambda x: x if x in ['Sleep Apnea', 'Insomnia'] else 'Normal')
        
        # Mapper les colonnes catégorielles
        df['BMI Category'] = df['BMI Category'].map(code_bmi).fillna(-1).astype(int)
        df['Gender'] = df['Gender'].map(code_gender).fillna(-1).astype(int)
        df['Occupation'] = df['Occupation'].map(code_occupation).fillna(-1).astype(int)
        df['Sleep Disorder'] = df['Sleep Disorder'].map(code_sleep_disorder).fillna(-1).astype(int)

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

        max_values = {'Gender': 2, 'Age': 130, 'Occupation': 5, 'Sleep Duration': 24, 
                      'Physical Activity Level': 200, 'Stress Level': 10, 'BMI Category': 4, 
                      'Blood Pressure': 200, 'Heart Rate': 200, 'Daily Steps': 50000, 'Quality of Sleep': 10}
       
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
        if on[0] == 'Quality of Sleep':
            max_value = 10
        return y / max_value
    
    def intern(df):
        df= encodage(df)
        df = imputation(df)

        for _ in on:
            X = df.drop(columns= on, axis=1)
        y = df[on[0]].values.reshape(-1, 1)

        # Normalize features
        X = normalisation(X)
        if y_normalisation:
            # Normalize target variable
            y = normalisation_y(y) 
        return X, y

    trainset, testset = split_data(df)
    X_train, y_train = intern(trainset)
    X_test, y_test = intern(testset)
    

    return X_train, y_train, X_test, y_test   

def preprocecing_user(df, on=None):
    """ Prétraite les données """

    def encodage(df):
        code_bmi = {'Normal': 0, 'Normal Weight': 0, 'Overweight': 1, 'Underweight': 2, 'Obese': 3}
        code_gender = {'Male': 0, 'Female': 1}
        code_occupation = {'working':0, 'unemployed':1, 'student':2, 'retired':3, 'other':4}

        df['Blood Pressure'] = df['Blood Pressure'].astype(str).str.split('/').str[0]
        df['Blood Pressure'] = pd.to_numeric(df['Blood Pressure'], errors='coerce')


        # Mapper les colonnes catégorielles
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
                      'Physical Activity Level': 200, 'Stress Level': 10, 'BMI Category': 4, 
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
        if on is not None:
            for col in on:
                if col in df.columns:
                    df.drop(columns=on, axis=1)
        else:
            df= encodage(df)
            df = imputation(df)
            df = normalisation(df) 
        return df

    return intern(df)
