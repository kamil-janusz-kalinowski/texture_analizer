from scripts.models import SVM_model, DecisionTree_model, LogicalRegression_model, NeuralNetwork_model, KNN_model, NaiveBayes_model, RandomForest_model, AdaBoost_model, GradientBoosting_model, XGBoost_model, LightGBM_model, CatBoost_model
from scripts.create_dataset import load_data_from_file
from sklearn.model_selection import train_test_split
import json

class Models_tester():
    def __init__(self, models, path_dataset, path_models_directory):
        self.models = models
        self.path_dataset = path_dataset
        self.path_models_directory = path_models_directory
        self.report = None
        
        X, Y = load_data_from_file(self.path_dataset)
        
        print('Data loaded from file')
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, shuffle=True)
        print('Data splitted')
        
    def train_models(self):
        print('Start training models')

        self.report = {}
        for ind, model in enumerate(self.models):
            print('-----------------------------------------------')
            print('Start training: ', model.name)
            model.fit(self.X_train, self.Y_train)
            report = model.get_report(self.X_test, self.Y_test)

            self.report[model.name] = {'report': report, 'time_fitting': model.time_fitting, 'time_prediction': model.time_predicting}
            
            model.save(f'{self.path_models_directory +'/'+ model.name}_model')
            
            print(f'{ind+1}/{len(self.models)} models trained and saved')
        
        print('All models trained and saved')
        
    def save_report(self, path_save='models/models_data.json'):
        with open(path_save, 'w') as f:
            json.dump(self.report, f, indent=2)
        print('Data saved to models_data.json')
        

# models = [
#     #SVM_model(), # SVM model is not included because it takes too long to train
#     DecisionTree_model(),
#     LogicalRegression_model(),
#     NeuralNetwork_model(),
#     KNN_model(),
#     NaiveBayes_model(),
#     RandomForest_model(),
#     AdaBoost_model(),
#     GradientBoosting_model(),
#     XGBoost_model(),
#     LightGBM_model(),
#     CatBoost_model()
# ]

path_dataset = r'datasets\dataset.pkl'
path_models_directory = r'models'
path_report = r'models/models_data.json'

# tester = Models_tester(models, path_dataset, path_models_directory)
# tester.train_models()
# tester.save_report(path_report)

# Display raports

class Report_manager():
    def __init__(self, path_report):
        self.path_report = path_report
        with open(self.path_report, 'r') as f:
            self.report = json.load(f)
            
    def display_report(self):
        # Sort models by f1-score
        models = sorted(self.report.items(), key=lambda x: x[1]['report']['weighted avg']['f1-score'], reverse=True)
        for model_name, data in models:
            print('-----------------------------------------------')
            print('Model: ', model_name)
            print('Time fitting: ', data['time_fitting'])
            print('Time predicting: ', data['time_prediction'])
            print('Report: ')
            print(data['report'])

    def display_f1_scores(self):
        models = sorted(self.report.items(), key=lambda x: x[1]['report']['weighted avg']['f1-score'], reverse=True)
        
        for model_name, data in models:
            print('-----------------------------------------------')
            print('Model: ', model_name)
            print('F1 score: ', data['report']['weighted avg']['f1-score'])
            
manager = Report_manager(path_report)
manager.display_f1_scores()

