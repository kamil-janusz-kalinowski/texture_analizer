import json
import logging

from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
import numpy as np
import pickle
from skimage import io


def load_data_from_file(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

class Features_extractor():
    """
    Class for extracting texture features from images using gray-level co-occurrence matrix (GLCM) analysis.
    """

    def __init__(self, distances=[1, 3, 5, 7], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], is_channels_independent=False):
        """
        Initialize the Features_extractor class.

        Parameters:
        - distances (list): List of distances for GLCM analysis.
        - angles (list): List of angles for GLCM analysis.
        - is_channels_independent (bool): Flag indicating whether to analyze each color channel independently.
        """
        self.distances = distances
        self.angles = angles
        self._is_channels_independent = is_channels_independent
        
    def save_extractor_parameters(self, path_save):
        """
        Save the parameters of the feature extractor to a JSON file.

        Parameters:
        - path_save (str): Path to save the JSON file.
        """
        data = {
            'distances': self.distances,
            'angles': self.angles,
            'is_channels_independent': self._is_channels_independent
        }
        
        with open(path_save, 'w') as f:
            json.dump(data, f)
            
    def load_extractor_parameters(self, path):
        """
        Load the parameters of the feature extractor from a JSON file.

        Parameters:
        - path (str): Path to the JSON file.
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.distances = data['distances']
        self.angles = data['angles']
        self._is_channels_independent = data['is_channels_independent']
        
    def get_texture_features(self, image):
        """
        Extract texture features from an image.

        Parameters:
        - image (numpy.ndarray): Input image.

        Returns:
        - features (dict): Dictionary of texture features extracted from the image.
        """
        if self._is_channels_independent:
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
    """
    A class that represents an image segmentator.

    Attributes:
        size_subsegment (int): The size of the subsegments.
        stride (int): The stride value for subsegment extraction.

    Methods:
        get_subsegments(image, boxes): Extracts subsegments from the given image based on the provided boxes.
    """

    def __init__(self, size_subsegment, stride):
        """
        Initializes an Image_segmentator object.

        Args:
            size_subsegment (int): The size of the subsegments.
            stride (int): The stride value for subsegment extraction.
        """
        self.size_subsegment = size_subsegment
        self.stride = stride
        
    def get_subsegments(self, image, boxes):
        """
        Extracts subsegments from the given image based on the provided boxes.

        Args:
            image (numpy.ndarray): The input image.
            boxes (list): A list of bounding boxes representing segments in the image.

        Returns:
            list: A list of subsegments extracted from the image.
        """
        segments = self._image2segments(image, boxes)
        subsegments = [self._segment2subsegments(segment) for segment in segments]
        
        return subsegments
    
    def _image2segments(self, image, boxes_segments):
        """
        Extracts segments from the given image based on the provided boxes.

        Args:
            image (numpy.ndarray): The input image.
            boxes_segments (list): A list of bounding boxes representing segments in the image.

        Returns:
            list: A list of segments extracted from the image.
        """
        segments = []
        for box in boxes_segments:
            (x1, y1), (x2, y2) = box
            segment = image[int(y1):int(y2), int(x1):int(x2)]
            segments.append(segment)
            
        return segments
    
    def _segment2subsegments(self, segment):
        """
        Extracts subsegments from the given segment.

        Args:
            segment (numpy.ndarray): The input segment.

        Returns:
            list: A list of subsegments extracted from the segment.
        """
        subsegments = []
        height, width = segment.shape[:2]

        for y in range(0, height - self.size_subsegment + 1, self.stride):
            for x in range(0, width - self.size_subsegment + 1, self.stride):
                subsegment = segment[y:y + self.size_subsegment, x:x + self.size_subsegment]
                subsegments.append(subsegment)
        
        return subsegments
    
class Image_analizer():
    """
    Class for analyzing texture features of an image using GLCM (Gray-Level Co-occurrence Matrix).
    
    Parameters:
    - distances (list): List of distances for GLCM calculation. Default is [1, 3, 5, 7].
    - angles (list): List of angles (in radians) for GLCM calculation. Default is [0, np.pi/4, np.pi/2, 3*np.pi/4].
    - is_channels_independent (bool): Flag indicating whether to analyze each color channel independently. Default is False.
    """
    def __init__(self, distances=[1, 3, 5, 7], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], is_channels_independent=False):
        self.distances = distances
        self.angles = angles
        self._is_channels_independent = is_channels_independent
        self._features = None
        
    def get_texture_features(self, image):
        """
        Calculate texture features of the given image.
        
        Parameters:
        - image: Input image to analyze.
        
        Returns:
        - features: Dictionary containing the calculated texture features.
        """
        if self._is_channels_independent:
            # Split RGB image to channels
            images = [self._preprocessing(image[:, :, ind]) for ind in range(3)]
        else:
            images = [self._preprocessing( rgb2gray(image) )]
            
        features = []
        for image in images:
            features = self._analize_texture(image)
        
        self._features = features
        
        return features
    
    def _preprocessing(self, image):
        """
        Preprocess the input image by normalizing it.
        
        Parameters:
        - image: Input image to preprocess.
        
        Returns:
        - image: Preprocessed image.
        """
        # Normalize image
        image = (image*255).astype('uint8')
        return image
            
    def _analize_texture(self, image):
        """
        Analyze the texture of the input image using GLCM.
        
        Parameters:
        - image: Input image to analyze.
        
        Returns:
        - properties: Dictionary containing the calculated texture properties.
        """
        glcm = graycomatrix(
            image, distances=self.distances, angles=self.angles, levels=256, symmetric=True, normed=True
        )

        params = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        properties = {prop: graycoprops(glcm, prop) for prop in params}

        return properties
    
    def make_features_flat(self, features=None):
        """
        Flatten the texture features dictionary into a 1D array.
        
        Parameters:
        - features: Optional input features dictionary. If not provided, the stored features will be used.
        
        Returns:
        - features_final: Flattened array of texture features.
        """
        if features is None:
            features = self._features
        # Flatten features
        features_flat = [value for key, value in features.items()]
        
        features_final = np.array(features_flat).reshape(-1)
        
        return features_final
    
class Dataset_creator():
    """
    A class for creating a dataset for texture analysis.

    Parameters:
    - path_annotation (str): The path to the annotation file.
    - path_save (str): The path to save the dataset (default: 'texture_training_data.pkl').
    """

    def __init__(self, path_annotation, path_save='texture_training_data.pkl'):
        self._annotation_reader = Annotation_reader(path_annotation)
        self.data_analizer = None
        self._path_save = path_save
        self._dataset = []
        
    def create_dataset(self, size_subsegment=32, stride=8, distances=[1, 3, 5, 7], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        """
        Creates the dataset using the specified parameters.

        Parameters:
        - size_subsegment (int): The size of the subsegments (default: 32).
        - stride (int): The stride for subsegmentation (default: 8).
        - distances (list): The distances for texture analysis (default: [1, 3, 5, 7]).
        - angles (list): The angles for texture analysis (default: [0, np.pi/4, np.pi/2, 3*np.pi/4]).
        """
        image_segmnetator = Image_segmentator(size_subsegment, stride)
        data_analizer = Image_analizer(distances, angles)

        labels = []
        features = []
        for path_image, boxes in self._annotation_reader.get_interator():
            offset_label = max(labels)+1 if labels else 0
            labels_temp = [ind + offset_label for ind in range(len(boxes))]
            
            image = io.imread(path_image)
            images_segmented = image_segmnetator.get_subsegments(image, boxes)
            for ind_zone, segment in enumerate(images_segmented):
                for subsegment in segment:
                    texture_features = data_analizer.get_texture_features(subsegment)
                    X = data_analizer.make_features_flat(texture_features)
                    features.append(X)
                    labels.append(labels_temp[ind_zone])
                    
        
        self._dataset = (np.array(features), np.array(labels))
        self.save_dataset()

    def save_dataset(self, path_save=None):
        """
        Saves the dataset to a file.

        Parameters:
        - path_save (str): The path to save the dataset (default: None, uses the path specified in the constructor).
        """
        if path_save is None:
            path_save = self._path_save
        with open(path_save, 'wb') as f:
            pickle.dump(self._dataset, f)
        logging.info(f"Dataset saved to {path_save}")

    