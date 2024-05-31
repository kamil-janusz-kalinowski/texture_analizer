
# ----------------- Create annotation ---------------------------------
from scripts.annotation import Annotator, AnnotatorSQL
import os

def get_all_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def create_annotation_SQL(path_database, paths_images):
    annot = AnnotatorSQL(path_database)
    for path_image in paths_images:
        annot.add_annotation(path_image)
    

path_database = "annotations\\annotation.db"
path_dir_images = "images"
paths_images = get_all_files(path_dir_images)

create_annotation_SQL(path_database, paths_images)

# ----------------------------- Create dataset ----------------------------------------------------------
from scripts.image_segmentator import ParamsImageSegmentator
from scripts.features_extractor import ParamsFeaturesExtractor
from scripts.create_dataset import DatasetCreator, DatabaseCreatorSQL

# def create_dataset(path_annotation, path_save_dataset, params_image_segmentator, params_features_extractor):

#     dataset_creator = DatasetCreator(path_annotation, path_save_dataset, params_image_segmentator, params_features_extractor)

#     dataset_creator.create_segments_from_annotation()
#     print("Done creating segments from annotation")

#     dataset_creator.create_subsegments_and_csv()
#     print("Done creating subsegments and csv")

#     dataset_creator.create_dataset_file()
#     print("Done creating dataset file")

# path_annotation = "annotations\\annotation.json"
# path_save_dataset = "datasets\\dataset_test"

# params_image_segmentator = ParamsImageSegmentator(32, 16)
# params_features_extractor = ParamsFeaturesExtractor()

# create_dataset(path_annotation, path_save_dataset, params_image_segmentator, params_features_extractor)


path_database = "datasets\\datasetSQL"
path_annotation = "annotations\\annotation.db"

params_image_segmentator = ParamsImageSegmentator(32, 16)
params_features_extractor = ParamsFeaturesExtractor()

# TODO: Test this!!!
def create_dataset_SQL(path_database, path_annotation, params_image_segmentator, params_features_extractor):

    database_creator = DatabaseCreatorSQL(path_database, path_annotation, params_image_segmentator, params_features_extractor)
    database_creator.create_all_dataset_parts()
    
    print("Done creating dataset from SQL")

create_dataset_SQL(path_database, path_annotation, params_image_segmentator, params_features_extractor)



