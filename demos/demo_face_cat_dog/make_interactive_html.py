"""A script to generate an interactive plotly html file.
The interactive plotly html file shows a set of images and their detection results by masking.
"""

from pathlib import Path
import json
import typing as ty
import copy
import toml

import math

import numpy as np

import logzero

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import plotly.graph_objects as go
import numpy as np
import plotly.express as px
from imageio import imread
import xarray as xr

import random

logger = logzero.logger


PATH_TOML_CONFIG = Path('./cat_and_dog_config/cat_and_dog_config.toml')
assert PATH_TOML_CONFIG.exists(), f'{PATH_TOML_CONFIG} does not exist'
dict_config = toml.load(PATH_TOML_CONFIG)

assert "base" in dict_config, f'base is not in {PATH_TOML_CONFIG}'
dataset_resource_config = dict_config["base"]

assert "path_experiment_root" in dataset_resource_config, f'path_experiment_root is not in {PATH_TOML_CONFIG}'
assert "dir_name_data" in dataset_resource_config, f'dir_name_data is not in {PATH_TOML_CONFIG}'

path_resource = Path(dataset_resource_config["path_experiment_root"])
assert path_resource.exists(), f'{path_resource} does not exist'

DICT_DETECTION_NAME2DETECTION_JSON = {
    'wasserstein-200': path_resource / "data" / "detection_output" / "wasserstein_independence.json",
    'cv-selection': path_resource / "data" / "detection_output" / "interpretable_mmd.json"
}

datasource_config = dict_config["data_setting"]
assert "path_dir_data_source_x" in datasource_config, f'dir_name_data is not in {PATH_TOML_CONFIG}'
assert "path_dir_data_source_y" in datasource_config, f'dir_name_data is not in {PATH_TOML_CONFIG}'


PATH_DIR_DOG_FACES = Path(datasource_config["path_dir_data_source_x"])
PATH_DIR_CAT_FACES = Path(datasource_config["path_dir_data_source_y"])
assert PATH_DIR_DOG_FACES.exists(), f'{PATH_DIR_DOG_FACES} does not exist'
assert PATH_DIR_CAT_FACES.exists(), f'{PATH_DIR_CAT_FACES} does not exist'

PATH_WORK_DIR = path_resource / "workdir_interactive_tool"

# If you want, Please update these varibales to your environment!
# DICT_DETECTION_NAME2DETECTION_JSON = {
#     'wasserstein-200': '/home/kmitsuzawa/DATA/mitsuzaw/eurecom/mmd-two-sample-test-variable-selection/cat_and_dogs/data/detection_output/wasserstein_independence.json',
#     # 'wasserstein-2000': '/media/DATA/mitsuzaw/mmd-tst-variable-detector/demos/cat-or-dog/detection_result_wasserstein_2000.json',
#     'cv-selection': '/home/kmitsuzawa/DATA/mitsuzaw/eurecom/mmd-two-sample-test-variable-selection/cat_and_dogs/data/detection_output/interpretable_mmd.json'
# }
# PATH_DIR_DOG_FACES = Path('/media/DATA/mitsuzaw/data-directory/animal_faces/afhq/train/cat')
# PATH_DIR_CAT_FACES = Path('/media/DATA/mitsuzaw/data-directory/animal_faces/afhq/train/dog')
# PATH_WORK_DIR = Path('/media/DATA/mitsuzaw/mmd-tst-variable-detector/demos/cat-or-dog/workdir_interactive_tool')


random.seed(10)

n_show_random_png = 15

# directory check
PATH_WORK_DIR.mkdir(parents=True, exist_ok=True)
assert PATH_DIR_DOG_FACES.exists(), f'{PATH_DIR_DOG_FACES} does not exist'
assert PATH_DIR_CAT_FACES.exists(), f'{PATH_DIR_CAT_FACES} does not exist'

# loading the detection json
logger.info('loading detection json')
dict_detection_obj = {}
image_pixel_size: int

__size_array = []

for __k, __p in DICT_DETECTION_NAME2DETECTION_JSON.items():
    assert Path(__p).exists(), f'{__p} does not exist'
    with open(__p, 'r') as f:
        # deciding the pixel size by the weights array in the object.
        __det_obj = json.load(f)
        
        _array_weight = np.array(__det_obj['detection_result']['weights'])
        __det_obj['array_weight'] = _array_weight
        __det_obj['variables'] = __det_obj['detection_result']['variables']
        __size_array.append(_array_weight.size)
        
        dict_detection_obj[__k] = __det_obj
    # end with
logger.info('loading detection json done')

assert len(set(__size_array)) == 1, f'weight array size is not same among all detection files. {set(__size_array)}'
image_pixel_size = __size_array[0]
# end for


# assert that all weight array is same among all detection files.
assert math.sqrt(image_pixel_size).is_integer(), f'image_pixel_size is not square number. {image_pixel_size}'
tuple_image_pixel_size = (int(math.sqrt(image_pixel_size)), int(math.sqrt(image_pixel_size)))
logger.info(f'image_pixel_size: {tuple_image_pixel_size}')

# creating heatmaps of the detection results.
# I pack these detection results into a single png file.
f_heatmap, ax_s_heat = plt.subplots(ncols=1, nrows=len(dict_detection_obj), figsize=(5, 5 * len(dict_detection_obj)))
__i_ax = 0
for __name_label, __det_obj in dict_detection_obj.items():
    __weighst_original_shape = np.reshape(__det_obj['array_weight'], tuple_image_pixel_size)
    __p_value_dev = __det_obj['detection_result']['p_value']
    sns.heatmap(__weighst_original_shape, ax=ax_s_heat[__i_ax], cmap='viridis')
    __label_message = f'{__name_label}\np-value={__p_value_dev}'
    ax_s_heat[__i_ax].set_title(__label_message)
    __i_ax += 1
# end for
plt.subplots_adjust(hspace=0.5)

__path_heatmap = PATH_WORK_DIR / 'heatmap.png'
f_heatmap.savefig(__path_heatmap.as_posix())
logger.info(f'heatmap saved at {__path_heatmap}')


# I create a set of masked images of the cat and dog faces.
# I randomly select a subset of images files.
# I compute a distance value with the selected images.
# I sort the images by the distance value.
seq_path_file_cat = list(sorted(PATH_DIR_CAT_FACES.rglob('*jpg')))
seq_path_file_dog = list(sorted(PATH_DIR_DOG_FACES.rglob('*jpg')))

n_iamegs = np.min([len(seq_path_file_cat), len(seq_path_file_dog)])
index_selection = random.sample(range(n_iamegs), k=n_show_random_png)

selected_path_file_cat = [__f for __i, __f in enumerate(seq_path_file_cat) if __i in index_selection]
selected_path_file_dog = [__f for __i, __f in enumerate(seq_path_file_dog) if __i in index_selection]

__path_dir_cat_or_dog = PATH_WORK_DIR / 'masked_images'
__path_dir_cat_or_dog.mkdir(parents=True, exist_ok=True)
logger.info(f'creating masked images at {__path_dir_cat_or_dog}')


def func_resize_image(path_file: Path, pixel_size: ty.Tuple[int, int]):
    im = Image.open(path_file)
    im = im.resize(pixel_size)
    return im
# end def

def func_overlay_detected_variables(path_file: Path, 
                                    detected_variables: ty.List[int],
                                    pixel_size: ty.Tuple[int, int]):
    im = Image.open(path_file)
    im = im.convert('L')
    im = im.resize(pixel_size)
    image_array_original = np.array(im)
    #
    image_array_overlay = copy.deepcopy(image_array_original)
    positions_discrepancy = np.unravel_index(detected_variables, pixel_size)
    for row_i, col_i in zip(*positions_discrepancy):
        image_array_overlay[row_i, col_i] = -1
    # end for
    
    return image_array_overlay
# end def


# stack_file_path = [__path_heatmap]
stack_file_path = []

__i_pair_image = 0
for path_file_cat, path_file_dog in zip(selected_path_file_cat, selected_path_file_dog):
    # The canvas is (n-detection-method + 1, n-dog-cat = 2)
    __f_cat_or_dog, __ax_s = plt.subplots(nrows=len(dict_detection_obj) + 1, ncols=2, figsize=(10, 5 * (len(dict_detection_obj) + 1)))
    __i_ax = 1
    
    # I resize the image to the same size as the detection result.
    __image_original_cat = func_resize_image(path_file_cat, tuple_image_pixel_size)
    __image_original_dog = func_resize_image(path_file_dog, tuple_image_pixel_size)    
    __ax_s[0, 0].imshow(__image_original_cat)
    __ax_s[0, 1].imshow(__image_original_dog)
    __ax_s[0, 0].set_title(path_file_cat.name)
    __ax_s[0, 1].set_title(path_file_dog.name)
    # I overlay the detected variables on the image.
    for __name_label, __det_obj in dict_detection_obj.items():
        image_cat = func_overlay_detected_variables(path_file_cat, __det_obj['variables'], tuple_image_pixel_size)
        image_dog = func_overlay_detected_variables(path_file_dog, __det_obj['variables'], tuple_image_pixel_size)
        __ax_s[__i_ax, 0].imshow(image_cat)
        __ax_s[__i_ax, 1].imshow(image_dog)
        
        __ax_s[__i_ax, 0].set_title('' * 10 + __name_label)
        __i_ax += 1
    # end for
    
    __path_mask = __path_dir_cat_or_dog / f'cat_or_dog_{__i_pair_image}.png'
    __f_cat_or_dog.savefig(__path_mask.as_posix())
    stack_file_path.append(__path_mask)
    logger.info(f'masked image saved at {__path_mask}')
    __i_pair_image += 1
# end for


def show_images_carousel(images, labels, key: str, title: str, height: int):        
    stacked = np.stack(images, axis=0)
    xrData = xr.DataArray(
        data   = stacked,
        dims   = [key, 'row', 'col', 'rgb'],
        coords = {key: labels}
    )
    # Hide the axes
    layout_dict = dict(yaxis_visible=False, 
                       yaxis_showticklabels=False, 
                       xaxis_visible=False, 
                       xaxis_showticklabels=False)
    fig = px.imshow(xrData, title=title, animation_frame=key).update_layout(layout_dict)
    fig.update_layout(
        autosize=False,
        width=images[0].shape[1] / 2,
        height=images[0].shape[0] / 2)
    return fig
# end def

# Show a list of URLs as a carousel, loading then as numpy images first
def show_images_carousel_from_urls(image_urls: ty.List[Path], labels, key: str, title: str, height: int):
    images = [imread(url, pilmode='RGB') for url in image_urls]
    return show_images_carousel(images, labels, key, title, height)
# end def

# main processes
images = {__p.as_posix(): __p.name for __p in stack_file_path}

fig = show_images_carousel_from_urls(images.keys(), list(images.values()), 'Method', None, 700)
path_html_plotly = PATH_WORK_DIR / 'interactive_plotly.html'
fig.write_html(path_html_plotly)
logger.info(f'plotly html saved at {path_html_plotly}')

# =============================================================