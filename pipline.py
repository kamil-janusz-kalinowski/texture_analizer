
# Create annotation for the image
from scripts.create_annotation import main as create_annotation

path_image = r"images\\Tiling_procedural_textures.jpg"  # Path to the image file
name_image = path_image.split("\\")[-1].split(".")[0]  # Extract the name of the image file
path_save = r"annotations\\" + name_image + ".json"  # Path to save the annotation file
create_annotation(path_image, path_save)

# Create dataset from the annotation
from scripts.create_dataset import main as create_dataset
from numpy import pi

path_annotation = path_save
path_save = r"datasets\texture_training_data.pkl"
size_subsegment = 32
stride = 10
distance = [1, 3, 5, 7]
angles = [0, pi/4, pi/2, 3*pi/4]
create_dataset(path_annotation, path_save, size_subsegment, stride, distance, angles)




