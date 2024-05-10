import json
import logging

from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import pickle
from skimage import io


def load_data_from_file(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

class Annotation_reader():
    def __init__(self, path_annotation):
        self._path_annotation = path_annotation
        self._data = self._read_annotation()
    
    def _read_annotation(self):
        with open(self._path_annotation, 'r') as json_file:
            data = json.load(json_file)
        return data
    
    def get_interator(self):
        with open(self._path_annotation, 'r') as json_file:
            data = json.load(json_file)
        
        for annotation in data['annotations']:
            path_image = annotation['filepath']
            boxes_segments = annotation['boxes']
            
            if not boxes_segments:
                continue
            
            yield path_image, boxes_segments
    
    def get_data(self):
        return self._data
    
    def get_image_paths(self):
        return [annotation['filepath'] for annotation in self._data['annotations']]
    
    def get_num_of_all_boxes(self):
        return sum([len(annotation['boxes']) for annotation in self._data['annotations']])
    
class Image_segmentator():
    def __init__(self, size_subsegment, stride):
        self.size_subsegment = size_subsegment
        self.stride = stride
        
    def get_subsegments(self, image, boxes):
        segments = self._image2segments(image, boxes)
        subsegments = [self._segment2subsegments(segment) for segment in segments]
        
        
        return subsegments
    
    def _image2segments(self, image, boxes_segments):
        segments = []
        for box in boxes_segments:
            (x1, y1), (x2, y2) = box
            segment = image[int(y1):int(y2), int(x1):int(x2)]
            segments.append(segment)
            
        return segments
    
    def _segment2subsegments(self, segment):
        subsegments = []
        height, width = segment.shape[:2]

        for y in range(0, height - self.size_subsegment + 1, self.stride):
            for x in range(0, width - self.size_subsegment + 1, self.stride):
                subsegment = segment[y:y + self.size_subsegment, x:x + self.size_subsegment]
                subsegments.append(subsegment)
        
        return subsegments
    
class Image_analizer():
    def __init__(self, distances=[1, 3, 5, 7], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], is_channels_independent=False):
        self.distances = distances
        self.angles = angles
        self._is_channels_independent = is_channels_independent
        self._features = None
        
    def get_texture_features(self, image):
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
        # Normalize image
        image = (image*255).astype('uint8')
        return image
            
    def _analize_texture(self, image):
        glcm = graycomatrix(
            image, distances=self.distances, angles=self.angles, levels=256, symmetric=True, normed=True
        )

        params = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        properties = {prop: graycoprops(glcm, prop) for prop in params}

        return properties
    
    def make_features_flat(self, features=None):
        if features is None:
            features = self._features
        # Flatten features
        features_flat = [value for key, value in features.items()]
        
        features_final = np.array(features_flat).reshape(-1)
        
        return features_final
    
class Dataset_creator():
    def __init__(self, path_annotation, path_save='texture_training_data.pkl'):
        self._annotation_reader = Annotation_reader(path_annotation)
        self.data_analizer = None
        self._path_save = path_save
        self._dataset = []
        
    def create_dataset(self,  size_subsegment=32, stride=8, distances=[1, 3, 5, 7], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
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
        if path_save is None:
            path_save = self._path_save
        with open(path_save, 'wb') as f:
            pickle.dump(self._dataset, f)
        logging.info(f"Dataset saved to {path_save}")

    