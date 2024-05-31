import os
from skimage import io
import matplotlib.pyplot as plt
from skimage.transform import resize
from tqdm import tqdm

def get_all_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

MIN_SIZE_OF_TEXTURE = 250

# Load all images from path_images
path_images = "images\\Archive"
path_save = "images"
paths_images = get_all_files(path_images)

for path_image in tqdm(paths_images,
                       desc="Resizing images",
                       unit="image",
                       total=len(paths_images)):
    
    # Load image
    image = io.imread(path_image)
    
    # Display image and get input from user about num of textures in x and y direction
    plt.imshow(image)
    plt.show()
    
    num_textures_x = int(input("Enter number of textures in x direction: "))
    num_textures_y = int(input("Enter number of textures in y direction: "))
    
    # Calculate size of texture
    size_texture_x = image.shape[1] // num_textures_x
    size_texture_y = image.shape[0] // num_textures_y
    
    # Get smaller size of texture
    size_texture = min(size_texture_x, size_texture_y)
    
    # Calc resize factor
    resize_factor = MIN_SIZE_OF_TEXTURE / size_texture
    
    # Resize image to size of texture
    image_resized = resize(image, (int(image.shape[0] * resize_factor), int(image.shape[1] * resize_factor)))
    image_resized = (255 * image_resized).astype("uint8") 
    
    # Save image   
    path_save_image = os.path.join(path_save, os.path.basename(path_image))
    io.imsave(path_save_image, image_resized)
    
