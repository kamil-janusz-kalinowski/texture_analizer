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
    
    def save_annotations(self, path_save=None):
        if path_save is not None:
            self._path_annotation = path_save
            
        with open(self._path_annotation, 'w') as f:
            json.dump(self._data, f, indent=2)
        print(f"Annotations saved to {self._path_annotation}")


if __name__ == "__main__":
    import os
    from scripts.create_dataset import get_all_files
    
    path_annotation_save = "annotation_test.json"
    path_dir_images = "images"
    paths_images = get_all_files(path_dir_images)

    annot = Annotator(path_annotation_save)

    for path_image in paths_images[0]:
        annot.add_annotation(path_image)
    annot.save_annotations()
    
    # Delete annotation_test.json after testing
    os.remove("annotation_test.json")