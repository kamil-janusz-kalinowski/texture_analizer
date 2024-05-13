from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
import time
import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class Model():
    """
    A class representing a machine learning model.

    Attributes:
        name (str): The name of the model.
        model: The underlying machine learning model.
        time_fitting (float): The time taken to fit the model.
        time_predicting (float): The time taken to make predictions using the model.
        report: The classification report generated by the model.

    Methods:
        fit(X, y): Fits the model to the given training data.
        predict(X): Makes predictions using the fitted model.
        get_report(X, y): Generates a classification report based on the model's predictions.
        save(path): Saves the model to a file.

    """

    def __init__(self):
        self.name = None
        self.model = None
        self.time_fitting = None
        self.time_predicting = None
        self.report = None


    def fit(self, X, y):
        """
        Fits the model to the given training data.

        Args:
            X: The input features.
            y: The target labels.

        Returns:
            self: The fitted model.

        """
        start = time.time()
        self.model.fit(X, y)
        self.time_fitting = time.time() - start
        return self

    def predict(self, X):
        """
        Makes predictions using the fitted model.

        Args:
            X: The input features.

        Returns:
            Y_pred: The predicted labels.

        """
        start = time.time()
        Y_pred = self.model.predict(X)
        self.time_fitting = time.time() - start
        return Y_pred

    def get_report(self, X, y):
        """
        Generates a classification report based on the model's predictions.

        Args:
            X: The input features.
            y: The true labels.

        Returns:
            report: The classification report.

        """
        start = time.time()
        Y_pred = self.predict(X)
        self.time_predicting = time.time() - start
        self.report = classification_report(y, Y_pred, output_dict=True)
        return self.report
    
    def save(self, path):
        """
        Saves the model to a file.

        Args:
            path (str): The path to save the model.

        """
        path_final = path + '.pkl'
        pickle.dump(self.model, open(path_final, 'wb'))
        print(f'Model saved to {path_final}')
        
def load_model(path) -> Model:
    """
    Load a trained model from the given path.

    Args:
        path (str): The path to the saved model file.

    Returns:
        Model: An instance of the Model class containing the loaded model.

    """
    model_obj = Model()
    model_obj.model = pickle.load(open(path, 'rb'))
    model_obj.name = path.split('/')[-1].split('.')[0]
    return model_obj

class SVM_model(Model):
    def __init__(self):
        super().__init__()
        self.name = 'SVM'
        self.model = SVC(kernel='linear', C=1, gamma='auto')

class DecisionTree_model(Model):
    def __init__(self):
        super().__init__()
        self.name = 'DecisionTree'
        self.model = DecisionTreeClassifier()

class LogicalRegression_model(Model):
    def __init__(self, max_iter=1000):
        super().__init__()
        self.name = 'LogicalRegression'
        self.model = LogisticRegression(max_iter=max_iter)
        
class NeuralNetwork_model(Model):
    """
    A class representing a neural network model.

    Parameters:
    - hidden_layer_sizes (tuple): The number of neurons in each hidden layer. Default is (100, 100).
    - max_iter (int): The maximum number of iterations. Default is 1000.
    """

    def __init__(self, hidden_layer_sizes=(100, 100), max_iter=1000):
        super().__init__()
        self.name = 'NeuralNetwork'
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
        
class KNN_model(Model):
    def __init__(self, n_neighbors = 3):
        super().__init__()
        self.name = 'KNN'
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        
class NaiveBayes_model(Model):
    def __init__(self):
        super().__init__()
        self.name = 'NaiveBayes'
        self.model = GaussianNB()
        
class RandomForest_model(Model):
    def __init__(self, n_estimators=100):
        super().__init__()
        self.name = 'RandomForest'
        self.model = RandomForestClassifier(n_estimators=n_estimators)
        
class AdaBoost_model(Model):
    def __init__(self, n_estimators=100):
        super().__init__()
        self.name = 'AdaBoost'
        self.model = AdaBoostClassifier(n_estimators=n_estimators)
        
class GradientBoosting_model(Model):
    def __init__(self, n_estimators=100):
        super().__init__()
        self.name = 'GradientBoosting'
        self.model = GradientBoostingClassifier(n_estimators=n_estimators)
        
class XGBoost_model(Model):
    def __init__(self):
        super().__init__()
        self.name = 'XGBoost'
        self.model = XGBClassifier()
        
class LightGBM_model(Model):
    def __init__(self):
        super().__init__()
        self.name = 'LightGBM'
        self.model = LGBMClassifier()
        
class CatBoost_model(Model):
    def __init__(self):
        super().__init__()
        self.name = 'CatBoost'
        self.model = CatBoostClassifier()
        
class SAMME_model(Model):
    def __init__(self):
        super().__init__()
        self.name = 'SAMME'
        self.model = AdaBoostClassifier(n_estimators=100, algorithm='SAMME')    
    
class Model_tester():
    def __init__(self, model, X_train, X_test, Y_train, Y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        
    def get_report(self):
        Y_pred = self.model.predict(self.X_test)
        report = classification_report(self.Y_test, Y_pred)
        return report
    
    def show_confusion_matrix(self):
        Y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.Y_test, Y_pred, normalize='pred')
        sns.heatmap(cm, annot=True)
        plt.show()
    
