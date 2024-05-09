
# Create annotation
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


# Create dataset
from numpy import pi
from scripts.create_dataset import Dataset_creator

path_annotation = "annotations\\annotation.json"
size_subsegment = 32
stride = 16
distance = [1, 3, 5, 7]
angles = [0, pi/4, pi/2, 3*pi/4]
#main(path_annotation, size_subsegment, stride, distance, angles)
dataset_creator = Dataset_creator(path_annotation)
dataset_creator.create_dataset(size_subsegment, stride, distance, angles)
dataset_creator.save_dataset()

