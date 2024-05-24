import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from skimage import io
import json

class Annotator():
    def __init__(self, path_annotation):
        self._path_annotation = path_annotation
        self._data = None
        self._load_annotations()
        
    def add_annotation(self, path_image):
        data_new = self._get_data_from_image(path_image)
        
        annotations = self._data['annotations']
        for annotation in annotations:
            if annotation['filepath'] == path_image:
                print("There is already an annotation for this image.")
                overwrite = input("Do you want to overwrite it? (y/n): ")
                if overwrite.lower() == 'y':
                    annotations.remove(annotation)
                    annotations.append(data_new)
                else:
                    return
        
        self._data['annotations'].append(data_new)

    def get_data(self):
        return self._data
    
    def _get_data_from_image(self, path_image):
        image = io.imread(path_image)
        
        data_annotation = {'filepath': path_image, 'boxes': []}
        
        def line_select_callback(eclick, erelease):
            'eclick and erelease are the press and release events'
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            data_annotation['boxes'].append(((x1, y1), (x2, y2)))
            print(f"({x1}, {y1}) --> ({x2}, {y2})")
            print(f" The button you used were: {eclick.button} {erelease.button}")

        def toggle_selector(event):
            print(' Key pressed.')
            if event.key in ['Q', 'q'] and toggle_selector.RS.active:
                print(' RectangleSelector deactivated.')
                toggle_selector.RS.set_active(False)
            if event.key in ['A', 'a'] and not toggle_selector.RS.active:
                print(' RectangleSelector activated.')
                toggle_selector.RS.set_active(True)

        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        ax.axis('off')

        toggle_selector.RS = RectangleSelector(ax, line_select_callback, useblit=True,
                                            button=[1, 3],  # don't use middle button
                                            minspanx=5, minspany=5,
                                            spancoords='pixels',
                                            interactive=True)
        fig.canvas.mpl_connect('key_press_event', toggle_selector)
        plt.show()
        
        return data_annotation

    def _load_annotations(self):
        # Try open JSON and if it doesnt exist create a new one
        try:
            with open(self._path_annotation, 'r') as f:
                self._data = json.load(f)
        except:
            self._data = {'annotations': []}
            with open(self._path_annotation, 'w') as f:
                json.dump(self._data, f)
    
    def save_to_json(self, path_save=None):
        if path_save is not None:
            self._path_annotation = path_save
            
        with open(self._path_annotation, 'w') as f:
            json.dump(self._data, f, indent=2)
        print(f"Annotations saved to {self._path_annotation}")

    
class AnnotationReader():
    """
    A class to read and process annotations from a JSON file.
    
    Parameters:
    path_annotation (str): The path to the JSON annotation file.
    
    Attributes:
    _path_annotation (str): The path to the JSON annotation file.
    _data (dict): The loaded annotation data.
    """
    
    def __init__(self, path_annotation):
        self._path_annotation = path_annotation
        self._data = self._read_annotation()
    
    def _read_annotation(self):
        """
        Read the annotation data from the JSON file.
        
        Returns:
        dict: The loaded annotation data.
        """
        with open(self._path_annotation, 'r') as json_file:
            data = json.load(json_file)
        return data
    
    def get_interator(self):
        """
        Get an iterator over the image paths and box segments in the annotation data.
        
        Yields:
        tuple: A tuple containing the image path and a list of box segments.
        """
        with open(self._path_annotation, 'r') as json_file:
            data = json.load(json_file)
        
        for annotation in data['annotations']:
            path_image = annotation['filepath']
            boxes_segments = annotation['boxes']
            
            if not boxes_segments:
                continue
            
            yield path_image, boxes_segments
    
    def get_data(self):
        """
        Get the loaded annotation data.
        
        Returns:
        dict: The loaded annotation data.
        """
        return self._data
    
    def get_image_paths(self):
        """
        Get a list of image paths from the annotation data.
        
        Returns:
        list: A list of image paths.
        """
        return [annotation['filepath'] for annotation in self._data['annotations']]
    
    def get_num_of_all_boxes(self):
        """
        Get the total number of boxes in the annotation data.
        
        Returns:
        int: The total number of boxes.
        """
        return sum([len(annotation['boxes']) for annotation in self._data['annotations']])

def get_all_files(path_dir):
    paths = []
    for root, dirs, files in os.walk(path_dir):
        for file in files:
            paths.append(os.path.join(root, file))
    return paths

if __name__ == "__main__":
    import os
    
    
    path_annotation_save = "annotation_test.json"
    path_dir_images = "images"
    paths_images = get_all_files(path_dir_images)
    paths_images = paths_images[:2]

    annot = Annotator(path_annotation_save)

    for path_image in paths_images:
        annot.add_annotation(path_image)
    
    data = annot.get_data()
    annot.save_to_json()
    
    # Delete annotation_test.json after testing
    os.remove("annotation_test.json")