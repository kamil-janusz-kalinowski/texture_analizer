import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from skimage import io
import json

def create_annotation(path_image, path_save):
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

    with open(path_save, 'w') as f:
        json.dump(data_annotation, f)

def main():
    # Example usage
    path_image = r"my_texture_analizer\images\\Tiling_procedural_textures.jpg"  # Path to the image file
    name_image = path_image.split("\\")[-1].split(".")[0]  # Extract the name of the image file
    
    path_save = r"my_texture_analizer\annotations\\" + name_image + ".json"  # Path to save the annotation file
    create_annotation(path_image, path_save)  # Call the create_annotation function with the image path and save path

    print('end of program')  # Print end of program message

if __name__ == "__main__":
    main()
