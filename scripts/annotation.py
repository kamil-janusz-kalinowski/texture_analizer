import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from skimage import io
from scripts.sql_database import TableManagerAnnotations

def get_all_files(path_dir):
    paths = []
    for root, dirs, files in os.walk(path_dir):
        for file in files:
            paths.append(os.path.join(root, file))
    return paths

class AnnotatorSQL():
    def __init__(self, path_database):
        self._path_database = path_database
        self._database = TableManagerAnnotations(path_database)
        
    def add_annotation(self, path_image):
        
        overwrite = None
        if path_image in self._database.get_all_image_paths():
            print("There is already an annotation for this image.")
            overwrite = input("Do you want to overwrite it? (y/n): ")
            if not overwrite.lower() == 'y':
                return
        
        boxes = self._select_box_from_image(path_image)
        if overwrite:
            self._database.delete_annotations_of_image(path_image)
        
        for box in boxes:
            self._database.add_record(path_image, box)
        
    def _select_box_from_image(self, path_image):
        image = io.imread(path_image)
        
        boxes = []
        
        def line_select_callback(eclick, erelease):
            'eclick and erelease are the press and release events'
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            boxes.append(((x1, y1), (x2, y2)))
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
        
        return boxes
    
    def __del__(self):
        self._database.close_connection()
        
    def __exit__(self, exc_type, exc_value, traceback):
        self._database.close_connection()        

class AnnotatorReaderSQL():
    def __init__(self, path_database):
        self._path_database = path_database
        self._database = TableManagerAnnotations(path_database)

    def get_interator(self):
        for record in self._database.get_all_records():
            yield record
            
    def get_all_data(self):
        return self._database.get_all_records()

    def get_num_of_all_boxes(self):
        return self._database.get_num_of_records()
    
    def get_num_of_records(self):
        return self._database.get_num_of_records()
    
    def get_image_paths(self):
        return self._database.get_all_image_paths()
    
    def __del__(self):
        self._database.close_connection()
        
    def __exit__(self, exc_type, exc_value, traceback):
        self._database.close_connection()
        


if __name__ == "__main__":
    import os
    
    path_annotation_save = "annotation_test.json"
    path_dir_images = "images"
    paths_images = get_all_files(path_dir_images)
    paths_images = paths_images[:2]
    
    # Class to delete the database file after the tests
    class DatabaseDeleter:
        def __init__(self, path_database, ann, ann_reader):
            self.path_database = path_database
            self.ann = ann
            self.ann_reader = ann_reader
            
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            del self.ann, self.ann_reader
            if os.path.exists(self.path_database):
                os.remove(self.path_database)
                

    path_database = "database_test.db"
    annot_sql = AnnotatorSQL(path_database)
    annot_reader_sql = AnnotatorReaderSQL(path_database)
    
    with DatabaseDeleter("database_test.db", annot_sql, annot_reader_sql):
        path_database = "database_test.db"
        annot_sql = AnnotatorSQL(path_database)
        annot_reader_sql = AnnotatorReaderSQL(path_database)
        
        for path_image in paths_images:
            annot_sql.add_annotation(path_image)

        print(annot_reader_sql.get_image_paths())
        print(annot_reader_sql.get_num_of_all_boxes())
        print(annot_reader_sql.get_all_data())
        
        
        
        
        

        
        
