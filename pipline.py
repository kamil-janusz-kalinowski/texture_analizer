
# Create annotation -----------------------------------------------------------------
from scripts.create_annotation import Annotator
import os

def get_all_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

path_annotation_save = "annotations\\annotation.json"
path_dir_images = "images"
paths_images = get_all_files(path_dir_images)

annot = Annotator(path_annotation_save)

for path_image in paths_images:
    annot.add_annotation(path_image)
annot.save_annotations()


# Create dataset -------------------------------------------------------------------
from scripts.create_dataset import Dataset_creator, Image_segmentator, Features_extractor
from numpy import pi

path_annotation = "annotations\\annotation.json"
path_save_dataset = "datasets\\dataset2"

params_image_segmentator = {"size_subsegment": 32, "stride": 24}
params_features_extractor = {"distance": [1, 3, 5, 7], "angles": [0, pi/4, pi/2, 3*pi/4]}

image_segmentator = Image_segmentator(params_image_segmentator["size_subsegment"], params_image_segmentator["stride"])
features_extractor = Features_extractor(params_features_extractor["distance"], params_features_extractor["angles"])

dataset_creator = Dataset_creator(path_annotation, path_save_dataset, image_segmentator, features_extractor)

dataset_creator.create_segments_from_annotation()
print("Done creating segments from annotation")

dataset_creator.create_subsegments_and_csv()
print("Done creating subsegments and csv")

dataset_creator.create_dataset_file()
print("Done creating dataset file")




