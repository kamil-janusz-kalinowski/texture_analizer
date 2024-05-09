import json
from skimage import io
from matplotlib import pyplot as plt
from skimage.color import rgb2gray

def show_segments(image, boxes_segments):

    for box in boxes_segments:
        (x1, y1), (x2, y2) = box
        segment = image[int(y1):int(y2), int(x1):int(x2)]
        io.imshow(segment)
        plt.axis('off')
        plt.title(f"Segment ({boxes_segments.index(box)+1}/{len(boxes_segments)})")
        plt.tight_layout()  # Adjust the position of the title
        io.show()

def get_data_from_annotation(path_json_file):
    # Open the JSON file and load the data
    with open(path_json_file, 'r') as json_file:
        data = json.load(json_file)

        # Load the image
        image_path = data['filepath']

        boxes_segments = data['boxes']

    return image_path, boxes_segments

def get_segments_from_annotation(path_json_file):
    # Open the JSON file and load the data
    path_image, boxes_segments = get_data_from_annotation(path_json_file)
    
    image = io.imread(path_image)

    segments = image2segments(image, boxes_segments)

    return segments

def image2segments(image, boxes_segments):
    segments = []
    for box in boxes_segments:
        (x1, y1), (x2, y2) = box
        segment = image[int(y1):int(y2), int(x1):int(x2)]
        segments.append(segment)
        
    return segments

def get_subsegments(segment, subsegment_size, stride=1):
    subsegments = []
    height, width = segment.shape[:2]

    for y in range(0, height - subsegment_size + 1, stride):
        for x in range(0, width - subsegment_size + 1, stride):
            subsegment = segment[y:y + subsegment_size, x:x + subsegment_size]
            subsegments.append(subsegment)

    return subsegments

def split_segments_to_subsegments(segments, subsegment_size: int, stride=1):
    subsegments = []
    labels = []
    for label, segment in enumerate(segments):
        subsegments_temp = get_subsegments(segment, subsegment_size, stride)
        subsegments.extend(subsegments_temp)
        labels.extend([label] * len(subsegments_temp))

    return subsegments, labels

def main(path_json_file=r'my_texture_analizer\annotations\Tiling_procedural_textures_annotation.json', size_subsegment=32, stride=10):
    # Example usage
    
    # Call the function with the specified JSON file path
    #show_segments_from_annotation(json_file_path)
    
    segments = get_segments_from_annotation(path_json_file)
    segments = [(rgb2gray(segment)*255).astype('uint8') for segment in segments] # Convert to grayscale
    
    print(f"Number of segments: {len(segments)}")
    
    subsegments, labels = split_segments_to_subsegments(segments, size_subsegment, stride)
    print(f"Number of subsegments: {len(subsegments)}")
    
    return subsegments, labels