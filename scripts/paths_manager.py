import os

class PathsManager():
    """
    A class that manages paths for a dataset.

    Args:
        path_dataset_main_folder (str): The path to the main folder of the dataset.

    Attributes:
        _path_main_folder (str): The path to the main folder of the dataset.
        _path_segments (str): The path to the segments folder.
        _path_subsegments (str): The path to the subsegments folder.
        _path_csv_segments (str): The path to the CSV file for segments categories.
        _path_csv_subsegments (str): The path to the CSV file for subsegments categories.
        _path_dataset_parameters (str): The path to the JSON file for dataset parameters.
        _path_input_output (str): The path to the pickle file for input/output.

    Methods:
        get_paths(): Returns a dictionary of all the paths.
        get_path(name): Returns the path corresponding to the given name.

    """

    def __init__(self, path_dataset_main_folder):
        self._path_main_folder = path_dataset_main_folder
        self._path_segments_folder = f"{self._path_main_folder}\\segments"
        self._path_subsegments_folder = f"{self._path_main_folder}\\subsegments"
        self._path_segments_csv = f"{self._path_main_folder}\\segments_categories.csv"
        self._path_subsegments_csv = f"{self._path_main_folder}\\subsegments_categories.csv"
        self._path_dataset_parameters = f"{self._path_main_folder}\\dataset_parameters.json"
        self._path_input_output_file = f"{self._path_main_folder}\\input_output.pkl"
        self._path_features_extractor_parameters = f"{self._path_main_folder}\\features_extractor_parameters.json"
        
    def get_paths(self):
        return {
            'path_main_folder': self._path_main_folder,
            'path_segments': self._path_segments_folder,
            'path_subsegments': self._path_subsegments_folder,
            'path_csv_segments': self._path_segments_csv,
            'path_csv_subsegments': self._path_subsegments_csv,
            'path_dataset_parameters': self._path_dataset_parameters,
            'path_input_output': self._path_input_output_file,
            'path_extractor_parameters': self._path_features_extractor_parameters
        }
    
    def get_main_folder(self):
        return self._path_main_folder
    
    def get_segments_folder(self):
        return self._path_segments_folder
    
    def get_subsegments_folder(self):
        return self._path_subsegments_folder
    
    def get_segments_csv(self):
        return self._path_segments_csv
    
    def get_subsegments_csv(self):
        return self._path_subsegments_csv
    
    def get_dataset_parameters(self):
        return self._path_dataset_parameters
    
    def get_input_output_file(self):
        return self._path_input_output_file
    
    def get_features_extractor_parameters(self):
        return self._path_features_extractor_parameters
    
    def create_folders(self):
        if not os.path.exists(self._path_main_folder):
            os.makedirs(self._path_main_folder)
        if not os.path.exists(self._path_segments_folder):
            os.makedirs(self._path_segments_folder)
        if not os.path.exists(self._path_subsegments_folder):
            os.makedirs(self._path_subsegments_folder)


if __name__ == "__main__":
    import shutil
    
    path_dataset = "dataset_test"
    paths_manager = PathsManager(path_dataset)
    paths = paths_manager.get_paths()
    print(paths)
    
    paths_manager.create_folders()
    
    assert os.path.exists(paths['path_main_folder'])
    assert os.path.exists(paths['path_segments'])
    assert os.path.exists(paths['path_subsegments'])
    
    print("Paths created successfully!")
    
    # Delete the folders created even if it not empty
    shutil.rmtree(paths_manager.get_main_folder())
    
    