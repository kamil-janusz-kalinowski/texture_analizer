
import logging
import warnings
import numpy as np
import pickle
from skimage import io
from skimage.color import gray2rgb

from scripts.paths_manager import PathsManager
from scripts.image_segmentator import ImageSegmentator, ParamsImageSegmentator
from scripts.features_extractor import FeaturesExtractor, ParamsFeaturesExtractor
from tqdm import tqdm
from scripts.annotation import AnnotatorReaderSQL
from scripts.sql_database import TextureDatabaseManager

class DatabaseCreatorSQL():
    def __init__(self, path_database, path_annotation_db, 
                 params_segmentor = ParamsImageSegmentator(), 
                 params_feature_extractor = ParamsFeaturesExtractor()):
        
        self._paths_manager = PathsManager(path_database)
        self._image_segmentator = ImageSegmentator(params_segmentor)
        self._features_extractor = FeaturesExtractor(params_feature_extractor)
        
        self._paths_manager.create_folders()
        
        self._database_manager = TextureDatabaseManager(self._paths_manager.path_database_file)
        
        if not path_annotation_db == self._paths_manager.path_database_file:
            self._copy_annotation_table(path_annotation_db)
            
        
    def create_all_dataset_parts(self):
        self.create_textures_table()
        self.create_segments_table()
        
    def _copy_annotation_table(self, path_annotation_db):
        annot = AnnotatorReaderSQL(path_annotation_db)
        for record in tqdm(annot.get_interator(), desc="Copying annotation table",
                                                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
                                                  ncols=75,
                                                    total=annot.get_num_of_records()):
            _ , path_image, boxes = record
            self._database_manager.table_annotation.add_record(path_image, boxes)
        logging.info(f"Annotation table copied to {self._paths_manager.path_database_file}")
    
    def create_textures_table(self):
        records = self._database_manager.table_annotation.get_all_records()
        for ind, path_image, box in tqdm(records, 
                                        desc="Creating textures from annotation",
                                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
                                        ncols=75,
                                        total=self._database_manager.table_annotation.get_num_of_records()):
            
            image = io.imread(path_image)
            (x1, y1), (x2, y2) = box
            segment = image[int(y1):int(y2), int(x1):int(x2)]
            
            path_save = f"{self._paths_manager.path_textures_folder}\\{ind}_{path_image.split('\\')[-1]}"
            io.imsave(path_save, segment)
            
            self._database_manager.table_texture.add_record(path_image, path_save, box)
                
        logging.info(f"Textures saved to {self._paths_manager.path_textures_folder}")
        
    def create_segments_table(self):
        
        for record in tqdm(self._database_manager.table_texture.get_all_records(),
                           desc="Creating segments from textures",
                           bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
                           ncols=75,
                           total=self._database_manager.table_texture.get_num_of_records()):
            ind_segment = 1
            ind_texture, _, path_texture, _ = record
            
            texture = io.imread(path_texture)
            boxes, image_parts = self._image_segmentator.get_image_part(texture)
            
            for box, segment in tqdm(zip(boxes, image_parts), 
                                    desc="Segmenting texture",
                                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
                                    ncols=75,
                                    total=len(boxes)):
                
                features = self._features_extractor.get_texture_features(segment)
                features_flat = self._features_extractor.make_features_flat(features)
                features_flat = list(features_flat) # Change it to list for SQL to save it as json
                
                path_save = f"{self._paths_manager.path_segments_folder}\\{ind_segment}_{path_texture.split('\\')[-1]}"
                
                # Turn off warning of imsave 
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    io.imsave(path_save, self._convert2rgb_uint8(segment))
                
                self._database_manager.table_subsegment.add_record(path_texture, path_save, box, features_flat, ind_texture)
                ind_segment += 1
                
        logging.info(f"Segments saved to {self._paths_manager.path_segments_folder}")

    def _convert2rgb_uint8(self, image):
        # Convert image to RGB format if it has only one color channel
        if image.ndim == 2:
            image = gray2rgb(image)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        
        if image.dtype == 'float64':
            image = image.astype(np.uint8)
        return image

            
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
    