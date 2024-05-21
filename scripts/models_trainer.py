import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import json

class Input_output_manager():
    def __init__(self, input = None, output = None, random_seed = None):
        self.input = input
        self.output = output
        self.random_seed = random_seed

    def load_from_pickle(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.input = data['input']
        self.output = data['output']
        
    def save_to_pickle(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'input': self.input, 'output': self.output}, f)
            
    def shuffle_data(self):
        p = np.random.permutation(len(self.input))
        self.input = self.input[p]
        self.output = self.output[p]
        
    def get_the_same_amount_of_data_from_each_category(self):
        # Set random seed if provided
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        # Get the same amount of data from each category
        num_of_categories = len(set(self.output))
        self.output = self.output.astype(int)
        min_amount = min([len(self.output[self.output == i]) for i in range(num_of_categories)])
        X_new = []
        Y_new = []
        for i in range(num_of_categories):
            category_data = self.input[self.output == i]
            category_labels = self.output[self.output == i]
            indices = np.random.choice(len(category_data), min_amount, replace=False)
            X_new.extend(category_data[indices])
            Y_new.extend(category_labels[indices])
        
        # Reset random seed if provided
        if self.random_seed is not None:
            np.random.seed(None)
        
        return np.array(X_new), np.array(Y_new)
    
    def get_input_output(self):
        return self.input, self.output

class Models_trainer():
    def __init__(self, models, path_input_output_file, path_models_directory):
        self.models = models
        self.path_input_output_file = path_input_output_file
        self.path_models_directory = path_models_directory
        self.input_output = Input_output_manager()
        self.report = None
        
        self.input_output.load_from_pickle(self.path_input_output_file)
        self.input_output.get_the_same_amount_of_data_from_each_category()
        self.input_output.shuffle_data()
        
        X, Y = self.input_output.get_input_output()
        
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
        
    def _get_the_same_amount_of_data_from_each_category(self, X, Y):
        # Get the same amount of data from each category
        num_of_categories = len(set(Y))
        Y = Y.astype(int)
        min_amount = min([len(Y[Y == i]) for i in range(num_of_categories)])
        X_new = []
        Y_new = []
        for i in range(num_of_categories):
            X_new.extend(X[Y == i][:min_amount])
            Y_new.extend(Y[Y == i][:min_amount])
        return np.array(X_new), np.array(Y_new)
    
    
    