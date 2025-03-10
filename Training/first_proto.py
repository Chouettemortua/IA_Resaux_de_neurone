import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import pickle
from tqdm import tqdm 


class Neurone: 

    def __init__(self, X=None, y=None, X_test=None, y_test=None, learning_rate=0.1, nb_iter=1, class_weights=None):
        self.W = None
        self.b = None
        self.L = []
        self.L_t = []
        self.acc = []
        self.acc_t = []
        self.class_weights = class_weights

        if X is not None and y is not None:
            self.W = np.random.randn(X.shape[1], 1)
            self.b = np.random.randn(1)
            print("Initial weights:", self.W)
            print("Initial bias:", self.b)
            self.train(X, y, X_test, y_test, learning_rate, nb_iter)

    def model(self, X):
        Z = X.dot(self.W) + self.b
        A = 1 / (1 + np.exp(-Z)) 
        return A

    def log_loss(self, A, y):
        epsilon = 1e-15
        if self.class_weights is not None:
            weights = np.array([self.class_weights[int(label[0])] for label in y])
            return np.sum(-weights * (y * np.log(A + epsilon) + (1 - y) * np.log(1 - A + epsilon))) / len(y)
        else:
            return np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon)) / len(y)

    def gradients(self, A, X, y):
        dW = np.dot(X.T, A - y) / len(y)
        db = np.sum(A - y) / len(y)
        return dW, db

    def update(self, dW, db, learning_rate):
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
    
    def preprocess_data(df):
        label_encoders = {}
        for column in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

        X = df.drop('Quality of Sleep', axis=1)
        y = df['Quality of Sleep']

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        y = y.values.reshape(-1, 1)

        # Check the distribution of the target variable
        print("Distribution of target variable:")
        print(pd.Series(y.flatten()).value_counts())

        return X, y, label_encoders, scaler
    
    def train_model(X_train, y_train, X_test, y_test):
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train.flatten())
        class_weights_dict = {i: class_weights[i] for i in np.unique(y_train)}
        print("Class weights:", class_weights_dict)

        if bool_c:
            sleep = Neurone(X_train, y_train, X_test, y_test, class_weights=class_weights_dict)
            sleep.save(path_n)
        else:
            sleep = Neurone(class_weights=class_weights_dict)
            sleep.load(path_n)

        if bool_t:
            sleep.train(X_train, y_train, X_test, y_test, 1e-2, 10000)
            sleep.save(path_n)
        return sleep

    df = load()
    X, y, label_encoders, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

