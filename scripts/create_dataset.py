
from scripts.segments_from_anotation import get_segments_from_annotation, split_segments_to_subsegments
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import pickle

def save_data_to_file(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
        
def load_data_from_file(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def show_dist_of_segments(result):
    distances = pdist(result, metric='euclidean')
    distances_matrix = squareform(distances)

    plt.imshow(distances_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    # Add title and labels
    plt.title('Distance Matrix')
    plt.xlabel('Segments')
    plt.ylabel('Segments')
    plt.show()

def analyze_segments(segments_gray, distances, angles):

    data = []
    for ind, segment in enumerate(segments_gray):

        glcm = graycomatrix(
            segment, distances=distances, angles=angles, levels=256, symmetric=True, normed=True
        )

        params = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        properties = {prop: graycoprops(glcm, prop) for prop in params}
        data.append(properties)
        print(f"Segment {ind+1}/{len(segments_gray)} processed")
        

    data_flat = [value.flatten() for element in data for key, value in element.items()]
    data_final = np.array(data_flat).reshape(len(data), -1)

    return data_final

def main(path_annotation, path_save='texture_training_data.pkl', size_subsegment=32, stride=10, distances=[1,3,5,7], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    segments = get_segments_from_annotation(path_annotation)
    print(f"Number of segments: {len(segments)}")

    segments = [(rgb2gray(segment)*255).astype('uint8') for segment in segments]

    subsegments, labels = split_segments_to_subsegments(segments, size_subsegment, stride)

    result = analyze_segments(subsegments, distances, angles)

    #show_dist_of_segments(result)
    
    save_data_to_file((result, labels), path_save)
    
    print("Data saved to file")


if __name__ == "__main__":
    # Example usage:
    path_annotation = r'my_texture_analizer\annotations\Tiling_procedural_textures_annotation.json'
    size_subsegment = 32
    stride = 10
    distance = [1, 3, 5, 7]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    main(path_annotation, size_subsegment, stride, distance, angles)