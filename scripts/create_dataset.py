import os
import pandas as pd
import logging
import warnings
import numpy as np
import pickle
from skimage import io
from skimage.color import gray2rgb

from scripts.paths_manager import PathsManager
from scripts.annotation import AnnotationReader
from scripts.image_segmentator import ImageSegmentator, ParamsImageSegmentator
from scripts.features_extractor import FeaturesExtractor, ParamsFeaturesExtractor

class InputOutputCreator():
    """
    A class that creates input and output files for a dataset.

    Args:
        path_csv_subsegments_categories (str): The path to the CSV file containing subsegment categories.

    Attributes:
        _path_csv_subsegments_categories (str): The path to the CSV file containing subsegment categories.
        _data (pandas.DataFrame): The loaded data from the CSV file.
        _analize_texture (Features_extractor): An instance of the Features_extractor class.

    Methods:
        _load_data(): Loads the data from the CSV file.
        create_input_output_file(): Creates the input and output files for the dataset.
        _save_file(path_save, data): Saves the data to a file.
        _get_input_and_output(): Extracts the input and output data from the loaded data.

    """

    def __init__(self, path_csv_subsegments_categories, 
                 params_feature_extractor = ParamsFeaturesExtractor()):
        
        self._path_csv_subsegments_categories = path_csv_subsegments_categories
        self._feature_extractor = FeaturesExtractor(params_feature_extractor)

    def _load_data_from_csv(self) -> pd.DataFrame:
        """
        Loads the data from the CSV file.

        Returns:
            pandas.DataFrame: The loaded data from the CSV file.

        """
        return pd.read_csv(self._path_csv_subsegments_categories)

    def create_input_output_file(self, save_path_file='dataset.pkl'):
        """
        Creates the input and output files for the dataset.

        """
        
        data = self._load_data_from_csv()
        X, Y = self._get_input_and_output(data)

        data_input_output = {'input': np.array(X), 'output': np.array(Y)}
        self._save_file(save_path_file, data_input_output)

    def _save_file(self, path_save, data):
        """
        Saves the data to a file.

        Args:
            path_save (str): The path to save the file.
            data: The data to be saved.

        """
        with open(path_save, 'wb') as f:
            pickle.dump(data, f)

    def _get_input_and_output(self, data: pd.DataFrame) -> tuple:
        """
        Extracts the input and output data from the loaded data.

        Returns:
            tuple: A tuple containing the input and output data.

        """
        X = []
        Y = []
        for ind, row in data.iterrows():
            image = io.imread(row['filename'])
            features = self._feature_extractor.get_texture_features(image)
            features_flat = self._feature_extractor.make_features_flat(features)

            X.append(features_flat)
            Y.append(row['category'])

        return X, Y
    
class DatasetCreator():
    def __init__(self, path_annotation, path_new_dataset, 
                 params_segmentor = ParamsImageSegmentator(), 
                 params_feature_extractor = ParamsFeaturesExtractor()):
        
        self._annotation_reader = AnnotationReader(path_annotation)
        self._image_segmentator = ImageSegmentator(params_segmentor)
        self._paths_manager = PathsManager(path_new_dataset)
        self._input_output_creator = InputOutputCreator(self._paths_manager.get_subsegments_csv() , params_feature_extractor)
        
        self._paths_manager.create_folders()
        
    def create_all_dataset_parts(self):
        self.create_segments_from_annotation()
        self.create_subsegments_and_csv()
        self.create_dataset_file()
        
    def create_segments_from_annotation(self):
        self._create_segments()
        self._create_csv_for_segments()
    
    def _create_segments(self):
        for path_image, boxes in self._annotation_reader.get_interator():
            image = io.imread(path_image)
            for ind_box, box in enumerate(boxes):
                (x1, y1), (x2, y2) = box
                segment = image[int(y1):int(y2), int(x1):int(x2)]
                
                path_save = f"{self._paths_manager.get_segments_folder()}\\{ind_box}_{path_image.split('\\')[-1]}"
                io.imsave(path_save, segment)
                
        logging.info(f"Segments saved to {self._paths_manager.get_segments_folder()}")
    
    def _create_csv_for_segments(self):
        files_names = os.listdir(self._paths_manager.get_segments_folder())
        data = []
        for category, filename in enumerate(files_names):
            data.append([filename, category])
        
        df = pd.DataFrame(data, columns=['filename', 'category'])
        df.to_csv(self._paths_manager.get_segments_csv(), index=False)
        logging.info(f"CSV file with segments categories saved to {self._paths_manager.get_segments_csv()}")
    
    def create_subsegments_and_csv(self):
        df = pd.read_csv(self._paths_manager.get_segments_csv())
        
        df_subsegments = pd.DataFrame(columns=['filename', 'category'])
        df_subsegments.to_csv(self._paths_manager.get_subsegments_csv(), index=False, mode='w')
        
        for ind, row in df.iterrows():
            path_image = f"{self._paths_manager.get_segments_folder()}\\{row['filename']}"
            category = row['category']
            image = io.imread(path_image)
            
            image_parts = self._image_segmentator.get_image_part(image)
            
            paths = []
            for ind_part, image_part in enumerate(image_parts):
                path_save = f"{self._paths_manager.get_subsegments_folder()}\\{category}_{ind_part}_{row['filename']}"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    io.imsave(path_save, self._convert2rgb_uint8(image_part))
                paths.append(path_save)
            
            df_new = pd.DataFrame({'filename': paths, 'category': [category]*len(paths)})
            df_new.to_csv(self._paths_manager.get_subsegments_csv(), index=False, mode='a', header=False)
    
    def _convert2rgb_uint8(self, image):
        # Convert image to RGB format if it has only one color channel
        if image.ndim == 2:
            image = gray2rgb(image)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        
        if image.dtype == 'float64':
            image = image.astype(np.uint8)
        return image
        
    def create_dataset_file(self):
        self._input_output_creator.create_input_output_file(self._paths_manager.get_input_output_file())
        logging.info(f"Input-output file saved to {self._paths_manager.get_input_output_file()}")    
            
            
def load_data_from_pickle_file(path_dataset_file) -> tuple:
    """
    Load data from a file.

    Args:
        path_dataset (str): The path to the dataset file.

    Returns:
        tuple: A tuple containing the input and output data.

    """
    with open(path_dataset_file, 'rb') as f:
        data = pickle.load(f)
    return data


# Example of usage
# if __name__ == "__main__":
#     import os
#     import logging
#     import shutil

#     logging.basicConfig(level=logging.INFO)
    
#     path_annotation = "data\\annotation.json"
#     path_new_dataset = "data\\dataset_test"
    
#     params_segmentator = ParamsImageSegmentator(size_subsegment=64, stride=64)
#     params_features_extractor = ParamsFeaturesExtractor(distances=[1, 3, 5, 7], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], is_extractor_gray=False)
    
#     dataset_creator = DatasetCreator(path_annotation, path_new_dataset, params_segmentator, params_features_extractor)
    
#     dataset_creator.create_all_dataset_parts()
    
#     print("Done")
    
#     # Delete dataset folder
#     shutil.rmtree(path_new_dataset)
    