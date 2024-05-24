
# ----------------- Create annotation ---------------------------------
from scripts.annotation import Annotator
import os

def get_all_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def create_annotation(path_annotation_save, paths_images):
    annot = Annotator(path_annotation_save)

    for path_image in paths_images:
        annot.add_annotation(path_image)
    annot.save_to_json()

path_annotation_save = "annotations\\annotation.json"
path_dir_images = "images"
paths_images = get_all_files(path_dir_images)

#create_annotation(path_annotation_save, paths_images)

# ----------------------------- Create dataset ----------------------------------------------------------
from scripts.image_segmentator import ParamsImageSegmentator
from scripts.features_extractor import ParamsFeaturesExtractor
from scripts.create_dataset import DatasetCreator

def create_dataset(path_annotation, path_save_dataset, params_image_segmentator, params_features_extractor):

    dataset_creator = DatasetCreator(path_annotation, path_save_dataset, params_image_segmentator, params_features_extractor)

    dataset_creator.create_segments_from_annotation()
    print("Done creating segments from annotation")

    dataset_creator.create_subsegments_and_csv()
    print("Done creating subsegments and csv")

    dataset_creator.create_dataset_file()
    print("Done creating dataset file")

path_annotation = "annotations\\annotation.json"
path_save_dataset = "datasets\\dataset_test"

params_image_segmentator = ParamsImageSegmentator(32, 16)
params_features_extractor = ParamsFeaturesExtractor()

create_dataset(path_annotation, path_save_dataset, params_image_segmentator, params_features_extractor)




