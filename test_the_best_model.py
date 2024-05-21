import json
from sklearn.model_selection import train_test_split
# Display raports

class Report_manager():
    def __init__(self, path_report):
        self.path_report = path_report
        with open(self.path_report, 'r') as f:
            self.report = json.load(f)
            
    def display_report(self):
        # Sort models by f1-score
        models = self._get_sorted_models()
        for model_name, data in models:
            print('-----------------------------------------------')
            print('Model: ', model_name)
            print('Time fitting: ', data['time_fitting'])
            print('Time predicting: ', data['time_prediction'])
            print('Report: ')
            print(data['report'])

    def display_f1_scores(self):
        models = self._get_sorted_models()
        
        for model_name, data in models:
            print('-----------------------------------------------')
            print('Model: ', model_name)
            print('F1 score: ', data['report']['weighted avg']['f1-score'])
            
    def get_report_of_the_best_model(self):
        models = self._get_sorted_models()
        best_model = models[0]
        return best_model
    
    def _get_sorted_models(self):
        models = sorted(self.report.items(), key=lambda x: x[1]['report']['weighted avg']['f1-score'], reverse=True)
        return models

path_report = r'models/models_data.json'
manager = Report_manager(path_report)
report = manager.get_report_of_the_best_model()
print('Best model: ', report[0])

# Load the best model -------------------------------------------------------------------------
from scripts.models import load_model
from scripts.create_dataset import load_data_from_pickle_file
from scripts.models import Model_tester

best_model = load_model(f'models/{report[0]}_model.pkl')

# Get dataset
X, Y = load_data_from_pickle_file(r'datasets/dataset.pkl')

# Split the dataset the same as in the training
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, shuffle=True)

tester = Model_tester(best_model, X_train, X_test, Y_train, Y_test)
report = tester.get_report()
print(report)
tester.show_confusion_matrix()

# Visualise results


