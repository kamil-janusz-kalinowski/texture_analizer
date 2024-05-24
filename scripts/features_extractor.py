import numpy as np
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops

class ParamsFeaturesExtractor():
    def __init__(self, distances=[1, 3, 5, 7], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], is_extractor_gray=False):
        self.distances = distances
        self.angles = angles
        self.is_extractor_gray = is_extractor_gray
        
    def get_parameters_of_extractor(self):
        return {
            'distances': self.distances,
            'angles': self.angles,
            'is_extractor_gray': self.is_extractor_gray
        }

class FeaturesExtractor():
    """
    Class for extracting texture features from images using gray-level co-occurrence matrix (GLCM) analysis.
    """

    def __init__(self, params = ParamsFeaturesExtractor()):
        """
        Initialize the Features_extractor class.

        Parameters:
        - distances (list): List of distances for GLCM analysis.
        - angles (list): List of angles for GLCM analysis.
        - is_channels_independent (bool): Flag indicating whether to analyze each color channel independently.
        """
        self.params = params
    
    def get_parameters_of_extractor(self):
        return self.params.get_parameters_of_extractor()

    def get_texture_features(self, image_temp: np.ndarray):
        """
        Extract texture features from an image.

        Parameters:
        - image (numpy.ndarray): Input image.

        Returns:
        - features (dict): Dictionary of texture features extracted from the image.
        """
        
        # Check if image is uint8
        if image_temp.dtype != 'uint8':
            raise ValueError("Image should be uint8")
        
        if self.params.is_extractor_gray:
            # Split RGB image to channels
            images = [image_temp[:, :, ind] for ind in range(3)]
        else:
            image = 255 * rgb2gray(image_temp)
            images = [image.astype(np.uint8)]
            
        features = []
        for image_temp in images:
            features = self._analyze_texture(image_temp)
        
        return features
    
    def _analyze_texture(self, image):
        """
        Analyze the texture of an image using gray-level co-occurrence matrix (GLCM) analysis.

        Parameters:
        - image (numpy.ndarray): Input image.

        Returns:
        - properties (dict): Dictionary of texture properties extracted from the image.
        """
        glcm = graycomatrix(
            image, distances=self.params.distances, angles=self.params.angles, levels=256, symmetric=True, normed=True
        )

        params = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        properties = {prop: graycoprops(glcm, prop) for prop in params}

        return properties

    def make_features_flat(self, features):
        """
        Flatten the extracted features.

        Parameters:
        - features (dict): Dictionary of texture features extracted from the image.

        Returns:
        - features_flat (numpy.ndarray): Flattened array of texture features.
        """
        return np.array([features[prop].flatten() for prop in features]).flatten()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    path_image = "images\\textures.png"
    image = plt.imread(path_image)
    image = (image * 255).astype(np.uint8)
    
    params = ParamsFeaturesExtractor()
    extractor = FeaturesExtractor(params)
    features = extractor.get_texture_features(image)
    
    print(features)
    print(extractor.make_features_flat(features))
    print(extractor.get_parameters_of_extractor())
    