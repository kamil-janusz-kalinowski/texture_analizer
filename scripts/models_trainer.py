import numpy as np
from scripts.sql_database import TableManagerSegments
from sklearn.model_selection import train_test_split
from scripts.models import Model
import pickle
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

class InputScaler():
    """
    A class for scaling input data using a specified scaler.

    Parameters:
    scaler : object, optional
        The scaler object to be used for scaling the input data. If not provided, a `StandardScaler` object will be used by default.

    Methods:
    transform(input)
        Scale the input data using the specified scaler.
    inverse_transform(input)
        Inverse scale the input data using the specified scaler.
    save(path)
        Save the scaler object to a file.
    load(path)
        Load the scaler object from a file.
    fit(input)
        Fit the scaler object to the input data.

    """

    def __init__(self, scaler:StandardScaler = None):
        self.scaler = scaler
            
    def transform(self, input) -> np.ndarray:
        """
        Scale the input data using the specified scaler.

        Parameters:
        input : array-like
            The input data to be scaled.

        Returns:
        array-like
            The scaled input data.
        """
        return self.scaler.transform(input)
    
    def inverse_transform(self, input) -> np.ndarray:
        """
        Inverse scale the input data using the specified scaler.

        Parameters:
        input : array-like
            The input data to be inverse scaled.

        Returns:
        array-like
            The inverse scaled input data.
        """
        return self.scaler.inverse_transform(input)
    
    def save(self, path) -> None:
        """
        Save the scaler object to a file.

        Parameters:
        path : str
            The path to save the scaler object.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
            
    def load(self, path) -> None:
        """
        Load the scaler object from a file.

        Parameters:
        path : str
            The path to load the scaler object from.
        """
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
            
    def fit(self, input) -> None:
        """
        Fit the scaler object to the input data.

        Parameters:
        input : array-like
            The input data to fit the scaler object to.
        """
        self.scaler = StandardScaler()
        self.scaler.fit(input)
        
class InputOutputSQLManager():
    def __init__(self, path_database):
        self._path_database = path_database
        self._table_segments = TableManagerSegments(self._path_database)
        self._data = self._table_segments.get_all_records()
        self.input = None
        self.output = None
        
    def load_input_output(self):
        X = []
        Y = []
        for record in self._data:
            X.append(record[4])
            Y.append(record[5])
            
        self.input = np.array(X)
        self.output = np.array(Y)
        
    def shuffle_data(self, seed = None):
        if seed is not None:
            np.random.seed(seed)
        p = np.random.permutation(len(self.input))
        self.input = self.input[p]
        self.output = self.output[p]
        if seed is not None:
            np.random.seed(None)
    
    def get_XY(self):
        return self.input, self.output
    
    def get_the_same_amount_of_data_from_each_category(self):
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
        return np.array(X_new), np.array(Y_new)
    
class ModelsTrainerSQL():
    def __init__(self, models: list[Model], path_database, path_models_directory, path_scaler_save = None):
        self.models = models
        self.path_database = path_database
        self.path_models_directory = path_models_directory
        self.input_output = InputOutputSQLManager(self.path_database)
        self.input_scaler = InputScaler()
        
        self.input_output.load_input_output()
        
        X, Y = self.input_output.get_XY()
        self.input_scaler.fit(X)
        
        if path_scaler_save:
            self.input_scaler.save(path_scaler_save)
        else:
            self.input_scaler.save(path_models_directory + 'input_scaler.pkl')

        
        self.input_output.get_the_same_amount_of_data_from_each_category()
        self.input_output.shuffle_data()
        
        X, Y = self.input_output.get_XY()
        X = self.input_scaler.transform(X)
        
        print('Data loaded from database')
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, shuffle=True)
        print('Data splitted')
        
    def train_and_save_models(self):
        print('Start training models')

        for ind, model in enumerate(self.models):
            print('-----------------------------------------------')
            print('Start training: ', model.name)
            model.fit(self.X_train, self.Y_train)
            model.get_report(self.X_test, self.Y_test)

            model.save(f'{self.path_models_directory +'/'+ model.name}_model')
            
            print(f'{ind+1}/{len(self.models)} models trained and saved')
        
        print('All models trained and saved')
        
        
    