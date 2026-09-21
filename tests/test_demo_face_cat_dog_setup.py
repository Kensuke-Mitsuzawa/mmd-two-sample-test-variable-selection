import tempfile
from pathlib import Path
import zipfile
import pytest

from demos.demo_face_cat_dog.setup_dataset import DatasetSetupExecutor, DatasetSetupResult


def test_resolve_dataset_paths():
    config_path = Path("/root/mmd-two-sample-test-variable-selection/demos/demo_face_cat_dog/cat_and_dog_config/cat_and_dog_config.toml")
    executor = DatasetSetupExecutor(path_file_config=config_path)
    dict_config = executor._load_config_toml(config_path)
    path_x, path_y, ext = executor._resolve_dataset_paths(dict_config)

    assert "train/cat" in str(path_x)
    assert "train/dog" in str(path_y)
    assert ext == "jpg"
# end def


def test_extract_archive_mock():
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        zip_path = tmp_dir / "test_afhq.zip"
        
        # Create a mock zip with afhq structure
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("afhq/train/cat/cat_001.jpg", b"fake_cat")
            zf.writestr("afhq/train/dog/dog_001.jpg", b"fake_dog")
        # end with

        target_x = tmp_dir / "data" / "afhq" / "train" / "cat"
        target_y = tmp_dir / "data" / "afhq" / "train" / "dog"

        executor = DatasetSetupExecutor(
            path_file_config=tmp_dir / "fake_config.toml"
        )
        executor._extract_archive_dataset(zip_path, target_x, target_y)

        assert (target_x / "cat_001.jpg").exists()
        assert (target_y / "dog_001.jpg").exists()
        assert executor._count_dataset_files(target_x, "jpg") == 1
        assert executor._count_dataset_files(target_y, "jpg") == 1
    # end with
# end def
