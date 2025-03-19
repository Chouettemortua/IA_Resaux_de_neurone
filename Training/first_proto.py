import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm 


class Neurone: 

    def __init__(self, X=None, y=None, X_test=None, y_test=None, learning_rate=0.1, nb_iter=1):
        self.W = None
        self.b = None
        self.L = []
        self.L_t = []
        self.acc = []
        self.acc_t = []

        if X is not None and y is not None:
            self.W = np.random.randn(X.shape[1], 1)
            self.b = np.random.randn(1)
            self.train(X, y, X_test, y_test, learning_rate, nb_iter)

    def model(self, X):
        Z = X.dot(self.W) + self.b
        A = 1 / (1 + np.exp(-Z)) 
        return A

    def log_loss(self, A, y):
        epsilon = 1e-15
        A = np.clip(A, epsilon, 1 - epsilon)  # Ensure A is within (0, 1)
        return -np.mean(y * np.log(A) + (1 - y) * np.log(1 - A))

    def gradients(self, A, X, y):
        dW = np.dot(X.T, A - y) / len(y)
        db = np.sum(A - y, axis=0) / len(y)
        return dW, db

    def update(self, dW, db, learning_rate):
        self.W -= learning_rate * dW
        self.b -= learning_rate * db

    def train(self, X, y, X_test, y_test, learning_rate=1e-2, nb_iter=10000, partialsteps=100):
        for i in tqdm(range(nb_iter)):
            A = self.model(X)

            if i % partialsteps == 0:
                self.L.append(self.log_loss(A, y))
                self.L_t.append(self.log_loss(self.model(X_test), y_test))
                self.acc.append(accuracy_score(y >= 0.5, self.predict(X)))
                self.acc_t.append(accuracy_score(y_test >= 0.5, self.predict(X_test)))

            dW, db = self.gradients(A, X, y)
            self.update(dW, db, learning_rate)

    def predict(self, X):
        A = self.model(X)
        return A >= 0.5

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({'W': self.W, 'b': self.b, 'L': self.L, 'L_t': self.L_t, 'acc': self.acc, 'acc_t': self.acc_t}, f)
        print(f"Modèle sauvegardé dans {filename}")

    def load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.W = data['W']
            self.b = data['b']
            self.L = data['L']
            self.L_t = data['L_t']
            self.acc = data['acc']
            self.acc_t = data['acc_t']
        print(f"Modèle chargé depuis {filename}")


class Resaux:
    def __init__(self):
        pass


def main_for_sleep_dat(bool_c, bool_t, path_n, path_c):

    def load():
        return pd.read_csv('Training/datasets/Sleep_health_and_lifestyle_dataset.csv')
    
    def encodage(df):

        code = {'Normal': 0, 'Normal Weight':0, 'Overweight': 1, 'Underweight':2, 'Obesity': 3, 'Software Eginneer': 0, 'Doctor': 1, 'Sales Representative': 2, 
                'Nurse': 3, 'Teacher': 4, 'Scientist': 5, 'Engineer': 6, 'Lawyer': 7, 'Accountant': 8, 'Salesperson': 9, 'Manager': 10,
                'Sleep Apnea': 1, 'Insomnia': 2, 'Male': 0, 'Female': 1 }
        

        df['Blood Pressure'] = df['Blood Pressure'].str.split('/').str[0].astype(int)
        df['Sleep Disorder'] = df['Sleep Disorder'].apply(lambda x: x if x in ['Sleep Apnea', 'Insomnia'] else 'Normal')
        
        for col in df.select_dtypes('object'):
            df[col] = df[col].map(code)

        return df

    def imputation(df):
        df = df.drop(columns=['Person ID'])
        return df.fillna(df.mean())
    
    def split_data(df):

        trainset, testset = train_test_split(df, test_size=0.2, random_state=12)
        return trainset, testset
    
    def normalisation(df):

        return df/df.max()
    
    def preprocecing(df):

        df = encodage(df)
        df = imputation(df)

        X = df.drop(columns='Quality of Sleep', axis=1)
        y = df['Quality of Sleep'].values.reshape(-1, 1)

        # Normalize features
        X = normalisation(X)
        y = normalisation(y) 

        return X, y
         
    def train_model(X_train, y_train, X_test, y_test):

        if bool_c:
            sleep = Neurone(X_train, y_train, X_test, y_test,)
            sleep.save(path_n)
        else:
            sleep = Neurone()
            sleep.load(path_n)

        initial_pred_train = sleep.predict(X_train)
        initial_pred_test = sleep.predict(X_test)
        print("Initial Train Accuracy:", accuracy_score(y_train >= 0.5, initial_pred_train))
        print("Initial Test Accuracy:", accuracy_score(y_test >= 0.5, initial_pred_test))

        if bool_t:
            sleep.train(X_train, y_train, X_test, y_test, 1e-2, 10000)
            sleep.save(path_n)
        return sleep


    data = load()
    df = data.copy()

    '''
    #visualisation des données

    print()
    print(df.head())
    print()
    print(df.info())
    print()
    print(df.describe())
    print()
    print(df.isna().sum()/df.shape[0])
    print()
    '''
    
    trainset, testset = split_data(df)
    X_train, y_train = preprocecing(trainset)
    X_test, y_test = preprocecing(testset)

    '''
    # test pour voir si les données sont bien prétraitées

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    print("\nX_train data types:\n", pd.DataFrame(X_train).dtypes)
    print("\nFirst 5 rows of X_train:\n", pd.DataFrame(X_train).head())
    print("\ny_train data types:\n", pd.DataFrame(y_train).dtypes)
    print("\nFirst 5 rows of y_train:\n", pd.DataFrame(y_train).head())
    '''
    
    sleep = train_model(X_train, y_train, X_test, y_test)

    y_pred_train = sleep.predict(X_train)
    y_pred_test = sleep.predict(X_test)


    print("Train Accuracy:", accuracy_score(y_train >= 0.5, y_pred_train))
    print("Test Accuracy:", accuracy_score(y_test >= 0.5, y_pred_test))

    # Additional metrics
    print("Train F1 Score:", f1_score(y_train >= 0.5, y_pred_train, average='weighted', zero_division= np.nan))
    print("Test F1 Score:", f1_score(y_test >= 0.5, y_pred_test, average='weighted', zero_division= np.nan))
    print("Train Precision:", precision_score(y_train >= 0.5, y_pred_train, average='weighted', zero_division= np.nan))
    print("Test Precision:", precision_score(y_test >= 0.5, y_pred_test, average='weighted', zero_division= np.nan))
    print("Train Recall:", recall_score(y_train >= 0.5, y_pred_train, average='weighted', zero_division= np.nan))
    print("Test Recall:", recall_score(y_test >= 0.5, y_pred_test, average='weighted', zero_division= np.nan))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(sleep.L, label="train loss")
    plt.plot(sleep.L_t, label="test loss")
    plt.legend()
    plt.title("Courbe de perte")
    plt.xlabel("Itérations")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(sleep.acc, label="train acc")
    plt.plot(sleep.acc_t, label="test acc")
    plt.legend()
    plt.title("Courbe d'accuracy")
    plt.xlabel("Itérations")
    plt.ylabel("Acc")

    plt.savefig(path_c)
    print("Courbes sauvegardée dans ", path_c)


if __name__ == "__main__":
    main_for_sleep_dat(True, True, "Training/saves/save_sleep.pkl", "Training/saves/curve_sleep.png")

