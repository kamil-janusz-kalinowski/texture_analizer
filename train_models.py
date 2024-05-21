from scripts.models import SVM_model, DecisionTree_model, LogicalRegression_model, NeuralNetwork_model, KNN_model, NaiveBayes_model, RandomForest_model, AdaBoost_model, GradientBoosting_model, XGBoost_model, LightGBM_model, CatBoost_model
from scripts.models_trainer import Models_trainer


# Models training and testing
models = [
    #SVM_model(), # SVM model is not included because it takes too long to train
    DecisionTree_model(),
    LogicalRegression_model(),
    NeuralNetwork_model(),
    KNN_model(),
    NaiveBayes_model(),
    RandomForest_model(),
    AdaBoost_model(),
    GradientBoosting_model(),
    XGBoost_model(),
    LightGBM_model(),
    CatBoost_model()
]

path_dataset = r'datasets\\dataset2\\input_output.pkl'
path_models_directory = r'models'
path_report = r'models\\models_data.json'

tester = Models_trainer(models, path_dataset, path_models_directory)
tester.train_models()
tester.save_report(path_report)

