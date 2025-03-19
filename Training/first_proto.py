import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
        return np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon), axis=0) / len(y)

    def gradients(self, A, X, y):
        #print(X.T.shape)
        #print(A.shape)
        #print(y.shape)
        #print((A - y).shape)
        dW = np.dot(X.T, A - y) / len(y)
        db = np.sum(A - y, axis = 0) / len(y)
        return dW, db

    def update(self, dW, db, learning_rate):
        #print(dW.shape)
        #print(self.W.shape)
        self.W -= learning_rate * dW
        self.b -= learning_rate * db

    def train(self, X, y, X_test, y_test, learning_rate=1e-2, nb_iter=100, partialsteps=10):
        for i in tqdm(range(nb_iter)):
            A = self.model(X)

            if i % partialsteps == 0:
                self.L.append(self.log_loss(A, y))
                self.L_t.append(self.log_loss(self.model(X_test), y_test))
                self.acc.append(accuracy_score(y, self.predict(X)))
                self.acc_t.append(accuracy_score(y_test, self.predict(X_test)))

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

        code = {'Normal': 0, 'Normal Weight':0, 'Overweight': 1, 'Underweight':2, 'Obesity': 2, 'Software Eginneer': 0, 'Doctor': 1, 'Sales Representative': 2, 
                'Nurse': 3, 'Teacher': 4, 'Scientist': 5, 'Engineer': 6, 'Lawyer': 7, 'Accountant': 8, 'Salesperson': 9, 'Manager': 10,
                'Sleep Apnea': 1, 'Insomnia': 2, 'Male': 0, 'Female': 1 }
        

        df['Blood Pressure'] = df['Blood Pressure'].str.split('/').str[0].astype(int)
        df['Sleep Disorder'] = df['Sleep Disorder'].apply(lambda x: x if x in ['Sleep Apnea', 'Insomnia'] else 'Normal')
        
        for col in df.select_dtypes('object'):
            df[col] = df[col].map(code)

        return df

    def imputation(df):

        return df.fillna(df.mean())
    
    def preprocecing(df):

        df = encodage(df)
        df = imputation(df)

        X = df.drop(columns='Quality of Sleep', axis=1)
        y = df['Quality of Sleep'].values.reshape(-1, 1)  # Reshape y to (299, 1)
        #print(y.shape)

        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X, y
         
    
    def train_model(X_train, y_train, X_test, y_test):

        if bool_c:
            sleep = Neurone(X_train, y_train, X_test, y_test,)
            sleep.save(path_n)
        else:
            sleep = Neurone()
            sleep.load(path_n)

        if bool_t:
            sleep.train(X_train, y_train, X_test, y_test, 1e-2, 10000)
            sleep.save(path_n)
        return sleep


    data = load()
    df = data.copy()

    """ visualisation des données
    print()
    print(df.head())
    print()
    print(df.info())
    print()
    print(df.describe())
    print()
    print(df.isna().sum()/df.shape[0])
    print()
    """
    
    trainset, testset = train_test_split(df, test_size=0.2, random_state=0)
    X_train, y_train = preprocecing(trainset)
    X_test, y_test = preprocecing(testset)

    """ test pour voir si les données sont bien prétraitées
    print()
    print(X_train.head())
    print()
    print(X_train.info())
    print()
    print(X_train.describe())
    print()
    print(X_train.isna().sum()/X_train.shape[0])
    print()
    """
    sleep = train_model(X_train, y_train, X_test, y_test)

    y_pred_train = sleep.predict(X_train)
    y_pred_test = sleep.predict(X_test)

    # Ensure y_train and y_test are in the correct shape for accuracy calculation
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # Debugging: Check initial predictions before training
    initial_pred_train = sleep.predict(X_train)
    initial_pred_test = sleep.predict(X_test)
    print("Initial Train Accuracy:", accuracy_score(y_train, initial_pred_train))
    print("Initial Test Accuracy:", accuracy_score(y_test, initial_pred_test))

    print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
    print("Test Accuracy:", accuracy_score(y_test, y_pred_test))

    # Additional metrics
    print("Train F1 Score:", f1_score(y_train, y_pred_train, average='weighted'))
    print("Test F1 Score:", f1_score(y_test, y_pred_test, average='weighted'))
    print("Train Precision:", precision_score(y_train, y_pred_train, average='weighted'))
    print("Test Precision:", precision_score(y_test, y_pred_test, average='weighted'))
    print("Train Recall:", recall_score(y_train, y_pred_train, average='weighted'))
    print("Test Recall:", recall_score(y_test, y_pred_test, average='weighted'))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(sleep.L, marker='+', label="train loss")
    plt.plot(sleep.L_t, marker='*', label="test loss")
    plt.legend()
    plt.title("Courbe de perte")
    plt.xlabel("Itérations")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(sleep.acc, marker='+', label="train acc")
    plt.plot(sleep.acc_t, marker='*', label="test acc")
    plt.legend()
    plt.title("Courbe d'accuracy")
    plt.xlabel("Itérations")
    plt.ylabel("Acc")

    plt.savefig(path_c)
    print("Courbes sauvegardée dans ", path_c)



if __name__ == "__main__":
    main_for_sleep_dat(True, True, "Training/saves/save_sleep.pkl", "Training/saves/curve_sleep.png")

