from scripts.create_dataset import load_data_from_file
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

def save_model(model, filename):
    pickle.dump(model, open(filename, 'wb'))
    print(f'Model saved to {filename}')
    
def load_model(filename):
    return pickle.load(open(filename, 'rb'))

def train_SVM_model(X_train, X_test, Y_train, Y_test):
    model = SVC(kernel='linear', C=1, gamma='auto')

    start = time.time()
    model.fit(X_train, Y_train)
    t = time.time() - start
    print(f"SVM: Training time: {t} seconds")

    print('SVM: Model trained')
    
    # Print classification report
    Y_pred = model.predict(X_test)
    raport = classification_report(Y_test, Y_pred)
    print(raport)
    return model, raport

def train_decision_tree_model(X_train, X_test, Y_train, Y_test):
    model = DecisionTreeClassifier()

    start = time.time()
    model.fit(X_train, Y_train)
    t = time.time() - start
    print(f"Decision Tree: Training time: {t} seconds")

    print('Decision Tree: Model trained')
    
    # Print classification report
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred))

    return model

def train_logical_regression_model(X_train, X_test, Y_train, Y_test, max_iter=1000):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=max_iter)

    start = time.time()
    model.fit(X_train, Y_train)
    t = time.time() - start
    print(f"Logical Regression: Training time: {t} seconds")
    
    # Print classification report
    Y_pred = model.predict(X_test)
    raport = classification_report(Y_test, Y_pred)
    print(raport)
    return model, raport

def train_neural_network_model(X_train, X_test, Y_train, Y_test, hidden_layer_sizes = (100, 100), max_iter = 1000):
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    raport = classification_report(Y_test, Y_pred)
    print(raport)
    return model, raport

def train_knn_model(X_train, X_test, Y_train, Y_test, n_neighbors):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    raport = classification_report(Y_test, Y_pred)
    print(raport)
    return model, raport
    
def train_naive_bayes_model(X_train, X_test, Y_train, Y_test):
    model = GaussianNB()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    raport = classification_report(Y_test, Y_pred)
    print(raport)
    return model, raport
    
def train_random_forest_model(X_train, X_test, Y_train, Y_test):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    raport = classification_report(Y_test, Y_pred)
    print(raport)
    return model, raport

def train_adaboost_model(X_train, X_test, Y_train, Y_test):
    model = AdaBoostClassifier(n_estimators=100) #TODO: check if it is possible to use DecisionTreeClassifier as base_estimator
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    raport = classification_report(Y_test, Y_pred)
    print(raport)
    return model, raport

def train_gradient_boosting_model(X_train, X_test, Y_train, Y_test):
    model = GradientBoostingClassifier(n_estimators=100)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    raport = classification_report(Y_test, Y_pred)
    print(raport)
    return model, raport

def train_xgboost_model(X_train, X_test, Y_train, Y_test):
    model = xgb.XGBClassifier()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    raport = classification_report(Y_test, Y_pred)
    print(raport)
    return model, raport

def train_lightgbm_model(X_train, X_test, Y_train, Y_test):
    model = lgb.LGBMClassifier()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    raport = classification_report(Y_test, Y_pred)
    print(raport)
    return model, raport

def train_catboost_model(X_train, X_test, Y_train, Y_test):
    model = CatBoostClassifier()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    raport = classification_report(Y_test, Y_pred)
    print(raport)
    return model, raport

def train_samme_model(X_train, X_test, Y_train, Y_test, max_depth=3, n_estimators=100):
    base_estimator = DecisionTreeClassifier(max_depth=max_depth)
    model = AdaBoostClassifier(estimator = base_estimator ,n_estimators=n_estimators, algorithm='SAMME')
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    raport = classification_report(Y_test, Y_pred)
    print(raport)
    return model, raport

def main():
    path_dataset = r'my_texture_analizer\datasets\texture_training_data.pkl'
    (X, Y) = load_data_from_file(path_dataset)

    print('Data loaded from file')

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, shuffle=True)
    print('Data splitted')

    # Dimension reduction
    # pca = PCA(n_components=30)
    # X_train_pca = pca.fit_transform(X_train) 
    # X_test_pca = pca.transform(X_test)

    # # SVM Classifier
    # model = train_SVM_model(X_train, X_test, Y_train, Y_test)
    # save_model(model, 'my_texture_analizer/models/SVM_model.pkl')
    
    # # Decision Tree Classifier
    # model = train_decision_tree_model(X_train, X_test, Y_train, Y_test)
    # save_model(model, 'my_texture_analizer/models/decision_tree_model.pkl')
    
    # # Logical regression
    # model = train_logical_regression_model(X_train, X_test, Y_train, Y_test)
    # save_model(model, 'my_texture_analizer/models/logical_regression_model.pkl')
    
    # # Neural Network
    # model = train_neural_network_model(X_train, X_test, Y_train, Y_test, (100, 100), 1000)
    # save_model(model, 'my_texture_analizer/models/neural_network_model.pkl')
    
    # # k-Nearest Neighbors
    # n_neighbors = len(set(Y))
    # train_knn_model(X_train, X_test, Y_train, Y_test, n_neighbors)
    # save_model(model, 'my_texture_analizer/models/knn_model.pkl')
    
    # # Naive Bayes
    # model = train_naive_bayes_model(X_train, X_test, Y_train, Y_test)
    # save_model(model, 'my_texture_analizer/models/naive_bayes_model.pkl')
    
    # # Random Forest
    # model = train_random_forest_model(X_train, X_test, Y_train, Y_test)
    # save_model(model, 'my_texture_analizer/models/random_forest_model.pkl')
    
    # # AdaBoost
    # model = train_adaboost_model(X_train, X_test, Y_train, Y_test)
    # save_model(model, 'my_texture_analizer/models/adaboost_model.pkl')
    
    # # Gradient Boosting
    # model = train_gradient_boosting_model(X_train, X_test, Y_train, Y_test)
    # save_model(model, 'my_texture_analizer/models/gradient_boosting_model.pkl')
    
    # # XGBoost
    # model = train_xgboost_model(X_train, X_test, Y_train, Y_test)
    # save_model(model, 'my_texture_analizer/models/xgboost_model.pkl')
    
    # # LightGBM
    # model = train_lightgbm_model(X_train, X_test, Y_train, Y_test)
    # save_model(model, 'my_texture_analizer/models/lightgbm_model.pkl')
    
    # # CatBoost
    # model = train_catboost_model(X_train, X_test, Y_train, Y_test)
    # save_model(model, 'my_texture_analizer/models/catboost_model.pkl')
    
    # # SAMME boosting
    # model = train_samme_model(X_train, X_test, Y_train, Y_test)
    # save_model(model, 'my_texture_analizer/models/samme_boosting_model.pkl')
    
    
    #TODO: Implement more models
    # Deep Learning Models
    
    # Convolutional Neural Network
    
    # Recurrent Neural Network
    
    # Ensemble methods
    
    # Factorization machines
    
    # AutoML
    
    
    print('All models trained and saved')
    

if __name__ == '__main__':
    main()