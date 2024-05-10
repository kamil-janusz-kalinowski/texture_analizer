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


class Model():
    def __init__(self):
        self.name = None
        self.model = None
        self.time_fitting = None
        self.time_predicting = None
        self.report = None


    def fit(self, X, y):
        start = time.time()
        self.model.fit(X, y)
        self.time_fitting = time.time() - start
        return self

    def predict(self, X):
        start = time.time()
        Y_pred = self.model.predict(X)
        self.time_fitting = time.time() - start
        return Y_pred

    def get_report(self, X, y):
        start = time.time()
        Y_pred = self.predict(X)
        self.time_predicting = time.time() - start
        self.report = classification_report(y, Y_pred, output_dict=True)
        return self.report
    
    def save(self, path):
        path_final = path + '.pkl'
        pickle.dump(self.model, open(path_final, 'wb'))
        print(f'Model saved to {path_final}')
        
def load_model(path):
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
    def __init__(self, hidden_layer_sizes = (100, 100), max_iter = 1000):
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
    
    
    