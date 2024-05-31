from scripts.models import DecisionTree, LogicalRegression_model, NeuralNetwork, KNN, NaiveBayes, RandomForest, AdaBoost, GradientBoosting, XGBoost, LightGBM, CatBoost
from scripts.models_trainer import ModelsTrainerSQL


# Models training and testing
models = [
    DecisionTree(),
    LogicalRegression_model(),
    NeuralNetwork(),
    KNN(),
    NaiveBayes(),
    RandomForest(),
    AdaBoost(),
    GradientBoosting(),
    XGBoost(),
    LightGBM(),
    CatBoost()
]

path_dataset = 'datasets\datasetSQL\database.db'
path_models_directory = r'models'

tester = ModelsTrainerSQL(models, path_dataset, path_models_directory)
tester.train_and_save_models()

