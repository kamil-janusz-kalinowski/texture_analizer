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
        get_main_folder(): Returns the path to the main folder of the dataset.
        get_segments_folder(): Returns the path to the segments folder.
        get_subsegments_folder(): Returns the path to the subsegments folder.
        get_segments_csv(): Returns the path to the CSV file for segments categories.
        get_subsegments_csv(): Returns the path to the CSV file for subsegments categories.
        get_dataset_parameters(): Returns the path to the JSON file for dataset parameters.
        get_input_output_file(): Returns the path to the pickle file for input/output.
        get_features_extractor_parameters(): Returns the path to the JSON file for features extractor parameters.
        create_folders(): Creates the necessary folders if they don't exist.

    """

    def __init__(self, path_dataset_main_folder):
        self.path_main_folder = path_dataset_main_folder
        self.path_textures_folder = f"{self.path_main_folder}\\segments"
        self.path_segments_folder = f"{self.path_main_folder}\\subsegments"
        
        self.path_database_file = f"{self.path_main_folder}\\database.db"
        
    def get_all_paths(self):
        """
        Returns a dictionary of all the paths.

        Returns:
            dict: A dictionary containing all the paths.

        """
        return {
            'path_main_folder': self.path_main_folder,
            'path_textures': self.path_textures_folder,
            'path_segments': self.path_segments_folder,
            'path_dataset_parameters': self.path_dataset_parameters,
            'path_extractor_parameters': self.path_features_extractor_parameters,
            'path_database_file': self._path_database_file
        }
    
    def create_folders(self):
        """
        Creates the necessary folders if they don't exist.

        """
        if not os.path.exists(self.path_main_folder):
            os.makedirs(self.path_main_folder)
        if not os.path.exists(self.path_textures_folder):
            os.makedirs(self.path_textures_folder)
        if not os.path.exists(self.path_segments_folder):
            os.makedirs(self.path_segments_folder)


if __name__ == "__main__":
    import shutil
    
    path_dataset = "dataset_test"
    paths_manager = PathsManager(path_dataset)
    paths = paths_manager.get_all_paths()
    print(paths)
    
    paths_manager.create_folders()
    
    assert os.path.exists(paths['path_main_folder'])
    assert os.path.exists(paths['path_segments'])
    assert os.path.exists(paths['path_subsegments'])
    
    print("Paths created successfully!")
    
    # Delete the folders created even if it not empty
    shutil.rmtree(paths_manager.get_main_folder())
    
    