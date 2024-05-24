from scripts.models import DecisionTree, LogicalRegression_model, NeuralNetwork, KNN, NaiveBayes, RandomForest, AdaBoost, GradientBoosting, XGBoost, LightGBM, CatBoost
from scripts.models_trainer import ModelsTrainer


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

path_dataset = r'datasets\\dataset2\\input_output.pkl'
path_models_directory = r'models'
path_report = r'models\\models_data.json'

tester = ModelsTrainer(models, path_dataset, path_models_directory)
tester.train_models()
tester.save_report(path_report)

