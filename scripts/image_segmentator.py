from skimage.transform import resize


class ParamsImageSegmentator():
    def __init__(self, size_subsegment = 32, stride = 16):
        self.size_subsegment = size_subsegment
        self.stride = stride
        
    def get_parameters_of_segmentator(self):
        return {
            'size_subsegment': self.size_subsegment,
            'stride': self.stride
        }

class ImageSegmentator():
    def __init__(self, params = ParamsImageSegmentator()):
        self.params = params
        
    def get_image_part(self, image):
        image_segments = []
        
        boxes = self._get_boxes(image.shape)
        for box in boxes:
            (x1, y1), (x2, y2) = box
            segment = image[y1:y2, x1:x2]
            image_segments.append(segment)
        
        return boxes, image_segments
    
    def _get_boxes(self, size_image):
        boxes = []
        for y in range(0, size_image[0] - self.params.size_subsegment + 1, self.params.stride):
            for x in range(0, size_image[1] - self.params.size_subsegment + 1, self.params.stride):
                box = [(x, y), (x + self.params.size_subsegment, y + self.params.size_subsegment)]
                boxes.append(box)
        
        return boxes

    
if __name__ == "__main__":
    from skimage import io
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Load Image
    path_image = "images\\textures.png"
    image = io.imread(path_image)
    
    # Create Image Segmentator
    params = ParamsImageSegmentator()
    segmentator = ImageSegmentator(params)
    
    # Get Image Part
    image_segments = segmentator.get_image_part(image)
    print(f"Number of image segments: {len(image_segments)}")
    print(f"Size of image segment: {image_segments[0].shape}")
    
    def display_image_segments(image_segments, size_display = (3, 6)):
        
        # Display image segments
        fig, ax = plt.subplots(size_display[0], size_display[1], figsize=(10, 5))
        
        for i in range(size_display[0]*size_display[1]):
            row = i // size_display[1]
            col = i % size_display[1]
            ax[row, col].imshow(image_segments[i].astype(np.uint8), cmap='gray')
            ax[row, col].axis('off')
            
        plt.show()
    
    display_image_segments(image_segments)
    