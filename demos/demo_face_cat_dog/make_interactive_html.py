"""A script to generate an interactive plotly html file.
The interactive plotly html file shows a set of images and their detection results by masking.
"""

from pathlib import Path
import argparse
import json
import typing as ty
import copy
import toml
import math
import random

import numpy as np
import logzero
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
import xarray as xr

logger = logzero.logger


def func_resize_image(path_file: Path, pixel_size: ty.Tuple[int, int]) -> Image.Image:
    """Resize image to target pixel size."""
    im = Image.open(path_file)
    im = im.resize(pixel_size)
    return im
# end def


def func_overlay_detected_variables(path_file: Path, 
                                    detected_variables: ty.List[int],
                                    pixel_size: ty.Tuple[int, int]) -> np.ndarray:
    """Overlay detected discrepancy variables on a grayscale version of the image."""
    im = Image.open(path_file)
    im = im.convert('L')
    im = im.resize(pixel_size)
    image_array_original = np.array(im)
    
    image_array_overlay = copy.deepcopy(image_array_original)
    positions_discrepancy = np.unravel_index(detected_variables, pixel_size)
    for row_i, col_i in zip(*positions_discrepancy):
        image_array_overlay[row_i, col_i] = -1
    # end for
    
    return image_array_overlay
# end def


def show_images_carousel(images: ty.List[np.ndarray], labels: ty.List[str], key: str, title: ty.Optional[str], height: int):        
    """Generate plotly carousel figure from list of images."""
    stacked = np.stack(images, axis=0)
    xrData = xr.DataArray(
        data=stacked,
        dims=[key, 'row', 'col', 'rgb'],
        coords={key: labels}
    )
    # Hide the axes
    layout_dict = dict(
        yaxis_visible=False, 
        yaxis_showticklabels=False, 
        xaxis_visible=False, 
        xaxis_showticklabels=False
    )
    fig = px.imshow(xrData, title=title, animation_frame=key).update_layout(layout_dict)
    fig.update_layout(
        autosize=False,
        width=images[0].shape[1] / 2,
        height=images[0].shape[0] / 2
    )
    return fig
# end def


def show_images_carousel_from_urls(image_urls: ty.Sequence[Path], labels: ty.List[str], key: str, title: ty.Optional[str], height: int):
    """Load image files as numpy RGB arrays and show as carousel."""
    images = [np.array(Image.open(url).convert('RGB')) for url in image_urls]
    return show_images_carousel(images, labels, key, title, height)
# end def


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate interactive plotly HTML visualization for detection results.")
    parser.add_argument(
        "--path_config",
        type=str,
        default=str(Path(__file__).parent / "cat_and_dog_config" / "cat_and_dog_config.toml"),
        help="Path to the TOML configuration file."
    )
    args = parser.parse_args()

    path_toml_config = Path(args.path_config)
    assert path_toml_config.exists(), f"{path_toml_config} does not exist"
    dict_config = toml.load(path_toml_config)

    assert "base" in dict_config, f"base section is not in {path_toml_config}"
    dataset_resource_config = dict_config["base"]

    assert "path_experiment_root" in dataset_resource_config, f"path_experiment_root is not in {path_toml_config}"
    path_resource = Path(dataset_resource_config["path_experiment_root"])
    assert path_resource.exists(), f"{path_resource} does not exist"

    datasource_config = dict_config["data_setting"]
    assert "path_dir_data_source_x" in datasource_config, f"path_dir_data_source_x is not in {path_toml_config}"
    assert "path_dir_data_source_y" in datasource_config, f"path_dir_data_source_y is not in {path_toml_config}"

    path_dir_dog_faces = Path(datasource_config["path_dir_data_source_x"])
    path_dir_cat_faces = Path(datasource_config["path_dir_data_source_y"])
    assert path_dir_dog_faces.exists(), f"{path_dir_dog_faces} does not exist"
    assert path_dir_cat_faces.exists(), f"{path_dir_cat_faces} does not exist"

    comp_config = dict_config.get("computational_resource", {})
    dir_name_detection_output = comp_config.get("dir_name_detection_output", "detection_output")
    path_detection_dir = path_resource / "data" / dir_name_detection_output
    if not path_detection_dir.exists():
        path_detection_dir = path_resource / dir_name_detection_output
    # end if

    path_work_dir = path_resource / "workdir_interactive_tool"
    path_work_dir.mkdir(parents=True, exist_ok=True)

    dict_detection_name2detection_json = {
        'wasserstein-200': path_detection_dir / "wasserstein_independence.json",
        'cv-selection': path_detection_dir / "interpretable_mmd.json"
    }

    # Filter to existing detection JSONs
    active_detection_json = {k: p for k, p in dict_detection_name2detection_json.items() if Path(p).exists()}
    if not active_detection_json:
        # Check any json files in detection output
        candidate_jsons = list(path_detection_dir.glob("*.json"))
        if not candidate_jsons:
            raise FileNotFoundError(
                f"No detection output json found in {path_detection_dir}. "
                f"Please run run_assessment.py first."
            )
        # end if
        for c_json in candidate_jsons:
            active_detection_json[c_json.stem] = c_json
        # end for
    # end if

    random.seed(10)
    n_show_random_png = 15

    logger.info("loading detection json")
    dict_detection_obj = {}
    size_array = []

    for k, p in active_detection_json.items():
        with open(p, 'r') as f:
            det_obj = json.load(f)
            array_weight = np.array(det_obj['detection_result']['weights'])
            det_obj['array_weight'] = array_weight
            det_obj['variables'] = det_obj['detection_result']['variables']
            size_array.append(array_weight.size)
            dict_detection_obj[k] = det_obj
        # end with
    # end for
    logger.info("loading detection json done")

    assert len(set(size_array)) == 1, f"weight array size is not same among all detection files: {set(size_array)}"
    image_pixel_size = size_array[0]

    assert math.sqrt(image_pixel_size).is_integer(), f"image_pixel_size is not square number: {image_pixel_size}"
    tuple_image_pixel_size = (int(math.sqrt(image_pixel_size)), int(math.sqrt(image_pixel_size)))
    logger.info(f"image_pixel_size: {tuple_image_pixel_size}")

    # Creating heatmaps of detection results
    n_methods = len(dict_detection_obj)
    f_heatmap, ax_s_heat = plt.subplots(
        ncols=1,
        nrows=n_methods,
        figsize=(5, 5 * n_methods),
        squeeze=False
    )
    for i_ax, (name_label, det_obj) in enumerate(dict_detection_obj.items()):
        weights_original_shape = np.reshape(det_obj['array_weight'], tuple_image_pixel_size)
        p_value_dev = det_obj['detection_result']['p_value']
        sns.heatmap(weights_original_shape, ax=ax_s_heat[i_ax, 0], cmap='viridis')
        label_message = f"{name_label}\np-value={p_value_dev}"
        ax_s_heat[i_ax, 0].set_title(label_message)
    # end for
    plt.subplots_adjust(hspace=0.5)

    path_heatmap = path_work_dir / 'heatmap.png'
    f_heatmap.savefig(path_heatmap.as_posix())
    logger.info(f"heatmap saved at {path_heatmap}")

    # Create masked images
    seq_path_file_cat = list(sorted(path_dir_cat_faces.rglob('*jpg')))
    seq_path_file_dog = list(sorted(path_dir_dog_faces.rglob('*jpg')))
    if not seq_path_file_cat or not seq_path_file_dog:
        # Also check png
        seq_path_file_cat = list(sorted(path_dir_cat_faces.rglob('*png')))
        seq_path_file_dog = list(sorted(path_dir_dog_faces.rglob('*png')))
    # end if

    n_images = min(len(seq_path_file_cat), len(seq_path_file_dog))
    assert n_images > 0, f"No image files found in {path_dir_cat_faces} or {path_dir_dog_faces}"
    k_sample = min(n_show_random_png, n_images)
    index_selection = random.sample(range(n_images), k=k_sample)

    selected_path_file_cat = [f for i, f in enumerate(seq_path_file_cat) if i in index_selection]
    selected_path_file_dog = [f for i, f in enumerate(seq_path_file_dog) if i in index_selection]

    path_dir_cat_or_dog = path_work_dir / 'masked_images'
    path_dir_cat_or_dog.mkdir(parents=True, exist_ok=True)
    logger.info(f"creating masked images at {path_dir_cat_or_dog}")

    stack_file_path = []
    for i_pair_image, (path_file_cat, path_file_dog) in enumerate(zip(selected_path_file_cat, selected_path_file_dog)):
        nrows = len(dict_detection_obj) + 1
        f_cat_or_dog, ax_s = plt.subplots(
            nrows=nrows,
            ncols=2,
            figsize=(10, 5 * nrows),
            squeeze=False
        )

        image_original_cat = func_resize_image(path_file_cat, tuple_image_pixel_size)
        image_original_dog = func_resize_image(path_file_dog, tuple_image_pixel_size)
        ax_s[0, 0].imshow(image_original_cat)
        ax_s[0, 1].imshow(image_original_dog)
        ax_s[0, 0].set_title(path_file_cat.name)
        ax_s[0, 1].set_title(path_file_dog.name)

        for i_method, (name_label, det_obj) in enumerate(dict_detection_obj.items()):
            image_cat = func_overlay_detected_variables(path_file_cat, det_obj['variables'], tuple_image_pixel_size)
            image_dog = func_overlay_detected_variables(path_file_dog, det_obj['variables'], tuple_image_pixel_size)
            ax_s[i_method + 1, 0].imshow(image_cat)
            ax_s[i_method + 1, 1].imshow(image_dog)
            ax_s[i_method + 1, 0].set_title(name_label)
            ax_s[i_method + 1, 1].set_title(name_label)
        # end for

        path_mask = path_dir_cat_or_dog / f"cat_or_dog_{i_pair_image}.png"
        f_cat_or_dog.savefig(path_mask.as_posix())
        plt.close(f_cat_or_dog)
        stack_file_path.append(path_mask)
        logger.info(f"masked image saved at {path_mask}")
    # end for

    images = {p.as_posix(): p.name for p in stack_file_path}
    fig = show_images_carousel_from_urls(list(images.keys()), list(images.values()), 'Method', None, 700)
    path_html_plotly = path_work_dir / 'interactive_plotly.html'
    fig.write_html(path_html_plotly)
    logger.info(f"plotly html saved at {path_html_plotly}")
# end def


if __name__ == "__main__":
    main()