import os
import pandas as pd
from skimage.transform import resize
import json
import logging

from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
import numpy as np
import pickle
from skimage import io
from skimage.transform import resize
from skimage.color import gray2rgb


class Paths_manager():
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

class Features_extractor():
    """
    Class for extracting texture features from images using gray-level co-occurrence matrix (GLCM) analysis.
    """

    def __init__(self, distances=[1, 3, 5, 7], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], is_extractor_gray=False):
        """
        Initialize the Features_extractor class.

        Parameters:
        - distances (list): List of distances for GLCM analysis.
        - angles (list): List of angles for GLCM analysis.
        - is_channels_independent (bool): Flag indicating whether to analyze each color channel independently.
        """
        self.distances = distances
        self.angles = angles
        self._is_extractor_gray = is_extractor_gray
    
    def get_parameters_of_extractor(self):
        return {
            'distances': self.distances,
            'angles': self.angles,
            'is_channels_independent': self._is_extractor_gray
        }

    def get_texture_features(self, image):
        """
        Extract texture features from an image.

        Parameters:
        - image (numpy.ndarray): Input image.

        Returns:
        - features (dict): Dictionary of texture features extracted from the image.
        """
        if self._is_extractor_gray:
            # Split RGB image to channels
            images = [self._preprocessing(image[:, :, ind]) for ind in range(3)]
        else:
            images = [self._preprocessing(rgb2gray(image))]
            
        features = []
        for image in images:
            features = self._analyze_texture(image)
        
        return features
    
    def _preprocessing(self, image):
        """
        Preprocess the image by normalizing its intensity values.

        Parameters:
        - image (numpy.ndarray): Input image.

        Returns:
        - image (numpy.ndarray): Preprocessed image.
        """
        # Normalize image
        image = (image * 255).astype('uint8')
        return image
    
    def _analyze_texture(self, image):
        """
        Analyze the texture of an image using gray-level co-occurrence matrix (GLCM) analysis.

        Parameters:
        - image (numpy.ndarray): Input image.

        Returns:
        - properties (dict): Dictionary of texture properties extracted from the image.
        """
        glcm = graycomatrix(
            image, distances=self.distances, angles=self.angles, levels=256, symmetric=True, normed=True
        )

        params = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        properties = {prop: graycoprops(glcm, prop) for prop in params}

        return properties

    def make_features_flat(self, features):
        """
        Flatten the extracted features.

        Parameters:
        - features (dict): Dictionary of texture features extracted from the image.

        Returns:
        - features_flat (numpy.ndarray): Flattened array of texture features.
        """
        return np.array([features[prop].flatten() for prop in features]).flatten()
    
class Annotation_reader():
    """
    A class to read and process annotations from a JSON file.
    
    Parameters:
    path_annotation (str): The path to the JSON annotation file.
    
    Attributes:
    _path_annotation (str): The path to the JSON annotation file.
    _data (dict): The loaded annotation data.
    """
    
    def __init__(self, path_annotation):
        self._path_annotation = path_annotation
        self._data = self._read_annotation()
    
    def _read_annotation(self):
        """
        Read the annotation data from the JSON file.
        
        Returns:
        dict: The loaded annotation data.
        """
        with open(self._path_annotation, 'r') as json_file:
            data = json.load(json_file)
        return data
    
    def get_interator(self):
        """
        Get an iterator over the image paths and box segments in the annotation data.
        
        Yields:
        tuple: A tuple containing the image path and a list of box segments.
        """
        with open(self._path_annotation, 'r') as json_file:
            data = json.load(json_file)
        
        for annotation in data['annotations']:
            path_image = annotation['filepath']
            boxes_segments = annotation['boxes']
            
            if not boxes_segments:
                continue
            
            yield path_image, boxes_segments
    
    def get_data(self):
        """
        Get the loaded annotation data.
        
        Returns:
        dict: The loaded annotation data.
        """
        return self._data
    
    def get_image_paths(self):
        """
        Get a list of image paths from the annotation data.
        
        Returns:
        list: A list of image paths.
        """
        return [annotation['filepath'] for annotation in self._data['annotations']]
    
    def get_num_of_all_boxes(self):
        """
        Get the total number of boxes in the annotation data.
        
        Returns:
        int: The total number of boxes.
        """
        return sum([len(annotation['boxes']) for annotation in self._data['annotations']])

class Image_segmentator():
    def __init__(self, size_subsegment, stride):
        self.size_subsegment = size_subsegment
        self.stride = stride
        
    def get_image_part(self, image):
        image_segments = []
        
        image = self._preprocessing(image)
        
        boxes = self._get_boxes(image.shape)
        for box in boxes:
            (x1, y1), (x2, y2) = box
            segment = image[y1:y2, x1:x2]
            image_segments.append(segment)
        
        return image_segments
    
    def _get_boxes(self, size_image):
        boxes = []
        for y in range(0, size_image[0] - self.size_subsegment + 1, self.stride):
            for x in range(0, size_image[1] - self.size_subsegment + 1, self.stride):
                box = [(x, y), (x + self.size_subsegment, y + self.size_subsegment)]
                boxes.append(box)
        
        return boxes
    
    def _preprocessing(self, image):
        # Change image resolution to have max 200 pixels in one dimension
        image = self._image_resolution_rescale(image)
        return image
    
    def _image_resolution_rescale(self, image, optimal_size=200):
        scale = optimal_size / min(image.shape[0:2])
        # Rescale image without losing the aspect ratio and color channels
        image = resize(image, (int(image.shape[0] * scale), int(image.shape[1] * scale)), preserve_range=True, anti_aliasing=True)
        return image

class Input_output_creator():
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

    def __init__(self, path_csv_subsegments_categories, features_extractor: Features_extractor):
        self._path_csv_subsegments_categories = path_csv_subsegments_categories
        self._feature_extractor = features_extractor

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

        data_input_output = (np.array(X), np.array(Y))
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
    
class Dataset_creator():
    def __init__(self, path_annotation, path_new_dataset, image_segmentator: Image_segmentator, features_extractor: Features_extractor):
        self._annotation_reader = Annotation_reader(path_annotation)
        self._image_segmentator = image_segmentator
        self._paths_manager = Paths_manager(path_new_dataset)
        self._input_output_creator = Input_output_creator(self._paths_manager.get_subsegments_csv() , features_extractor)
        
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
               