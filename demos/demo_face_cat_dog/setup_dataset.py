#!/usr/bin/env python3
"""Script to set up the AFHQ cat and dog face dataset.

This script loads the experiment configuration file (.toml), creates the target
directories, and downloads and extracts the AFHQ dataset following the official
StarGAN v2 dataset specifications.
"""

from pathlib import Path
import argparse
import sys
import typing as ty
import urllib.request
import zipfile
import shutil

from pydantic import BaseModel, Field
import toml
from tqdm import tqdm
import logzero
from logzero import logger


DEFAULT_AFHQ_URL = "https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=1"


class DatasetSetupResult(BaseModel):
    """Result summary of the dataset setup process."""
    path_dir_x: Path = Field(description="Directory path of dataset X (e.g. cat)")
    path_dir_y: Path = Field(description="Directory path of dataset Y (e.g. dog)")
    count_files_x: int = Field(description="Number of image files in dataset X")
    count_files_y: int = Field(description="Number of image files in dataset Y")
    is_success: bool = Field(description="Whether the dataset setup succeeded")
# end class


class DatasetSetupExecutor:
    """Executes the dataset download and directory configuration."""

    def __init__(
        self,
        path_file_config: Path,
        url_dataset_archive: str = DEFAULT_AFHQ_URL,
        is_force_download: bool = False,
        is_clean_zip: bool = False,
    ) -> None:
        self.path_file_config = path_file_config.resolve()
        self.url_dataset_archive = url_dataset_archive
        self.is_force_download = is_force_download
        self.is_clean_zip = is_clean_zip
    # end def

    def execute_setup_dataset(self) -> DatasetSetupResult:
        """Main method to execute the setup procedure."""
        logger.info(f"Loading configuration from: {self.path_file_config}")
        dict_config = self._load_config_toml(self.path_file_config)
        
        path_dir_x, path_dir_y, file_extension = self._resolve_dataset_paths(dict_config)
        logger.info(f"Target dataset X path: {path_dir_x}")
        logger.info(f"Target dataset Y path: {path_dir_y}")

        # Check if dataset is already set up and contains files
        count_x = self._count_dataset_files(path_dir_x, file_extension)
        count_y = self._count_dataset_files(path_dir_y, file_extension)

        if count_x > 0 and count_y > 0 and not self.is_force_download:
            logger.info(
                f"Dataset already exists with {count_x} files in X and {count_y} files in Y. "
                "Skipping download (use --force to re-download)."
            )
            return DatasetSetupResult(
                path_dir_x=path_dir_x,
                path_dir_y=path_dir_y,
                count_files_x=count_x,
                count_files_y=count_y,
                is_success=True,
            )
        # end if

        # Determine download directory and zip destination
        path_dir_download = self._determine_download_directory(path_dir_x)
        path_dir_download.mkdir(parents=True, exist_ok=True)
        path_file_zip = path_dir_download / "afhq.zip"

        if not path_file_zip.exists() or self.is_force_download:
            logger.info(f"Downloading dataset archive from {self.url_dataset_archive}...")
            self._download_archive_file(self.url_dataset_archive, path_file_zip)
        else:
            logger.info(f"Archive file {path_file_zip} already exists. Skipping download.")
        # end if

        logger.info(f"Extracting archive {path_file_zip}...")
        self._extract_archive_dataset(path_file_zip, path_dir_x, path_dir_y)

        if self.is_clean_zip:
            logger.info(f"Removing archive file: {path_file_zip}")
            path_file_zip.unlink(missing_ok=True)
        # end if

        count_x = self._count_dataset_files(path_dir_x, file_extension)
        count_y = self._count_dataset_files(path_dir_y, file_extension)

        if count_x == 0 or count_y == 0:
            raise FileNotFoundError(
                f"Dataset setup incomplete: {count_x} files in {path_dir_x}, {count_y} files in {path_dir_y}"
            )
        # end if

        logger.info(
            f"Dataset setup completed successfully: {count_x} files in {path_dir_x}, {count_y} files in {path_dir_y}"
        )
        return DatasetSetupResult(
            path_dir_x=path_dir_x,
            path_dir_y=path_dir_y,
            count_files_x=count_x,
            count_files_y=count_y,
            is_success=True,
        )
    # end def

    def _load_config_toml(self, path_toml: Path) -> ty.Dict[str, ty.Any]:
        """Load and parse the TOML configuration file."""
        if not path_toml.exists():
            raise FileNotFoundError(f"Configuration file not found: {path_toml}")
        # end if
        with open(path_toml, "r") as file_handle:
            config_data = toml.load(file_handle)
        # end with
        return config_data
    # end def

    def _resolve_dataset_paths(
        self,
        dict_config: ty.Dict[str, ty.Any]
    ) -> ty.Tuple[Path, Path, str]:
        """Resolve dataset source directory paths and file extension."""
        if "data_setting" not in dict_config:
            raise KeyError("Missing 'data_setting' section in TOML config.")
        # end if
        data_setting = dict_config["data_setting"]

        if "path_dir_data_source_x" not in data_setting or "path_dir_data_source_y" not in data_setting:
            raise KeyError(
                "Missing 'path_dir_data_source_x' or 'path_dir_data_source_y' in 'data_setting' config."
            )
        # end if

        path_x = Path(data_setting["path_dir_data_source_x"])
        path_y = Path(data_setting["path_dir_data_source_y"])
        file_extension = str(data_setting.get("file_extension", "jpg")).lstrip(".")

        # If relative, resolve relative to the demo folder (parent of config folder)
        path_base_demo = self.path_file_config.parent.parent
        if not path_x.is_absolute():
            path_x = (path_base_demo / path_x).resolve()
        # end if
        if not path_y.is_absolute():
            path_y = (path_base_demo / path_y).resolve()
        # end if

        return path_x, path_y, file_extension
    # end def

    def _determine_download_directory(self, path_dir_x: Path) -> Path:
        """Determine suitable parent directory for dataset download."""
        # path_dir_x is typically <...>/data/afhq/train/cat
        # We want to place afhq.zip in <...>/data/
        current_path = path_dir_x
        while current_path.parent != current_path:
            if current_path.name in ("afhq", "train", "cat", "dog"):
                current_path = current_path.parent
            else:
                break
            # end if
        # end while
        return current_path
    # end def

    def _count_dataset_files(self, path_dir: Path, file_extension: str) -> int:
        """Count existing image files matching the extension in the directory."""
        if not path_dir.exists() or not path_dir.is_dir():
            return 0
        # end if
        files = list(path_dir.rglob(f"*.{file_extension}"))
        return len(files)
    # end def

    def _download_archive_file(self, url_download: str, path_output_file: Path) -> None:
        """Stream-download file with progress bar."""
        request_obj = urllib.request.Request(
            url_download,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(request_obj) as response_stream:
            total_size_bytes = int(response_stream.headers.get("Content-Length", 0))
            block_size_bytes = 1024 * 1024  # 1 MB chunk

            with open(path_output_file, "wb") as file_out:
                with tqdm(
                    total=total_size_bytes,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=path_output_file.name,
                ) as progress_bar:
                    while True:
                        buffer = response_stream.read(block_size_bytes)
                        if not buffer:
                            break
                        # end if
                        file_out.write(buffer)
                        progress_bar.update(len(buffer))
                    # end while
                # end with
            # end with
        # end with
    # end def

    def _extract_archive_dataset(
        self,
        path_file_zip: Path,
        path_dir_target_x: Path,
        path_dir_target_y: Path,
    ) -> None:
        """Extract zip archive and ensure destination directory structure matches config."""
        with zipfile.ZipFile(path_file_zip, "r") as zip_ref:
            all_names = zip_ref.namelist()
            logger.info(f"Archive contains {len(all_names)} entries.")

            # Check if archive root has 'afhq/' or starts directly with 'train/'
            has_afhq_root = any(name.startswith("afhq/") for name in all_names)

            # Target destination for extraction
            # If path_dir_target_x ends in .../afhq/train/cat:
            # - if has_afhq_root: extract to parent of 'afhq' (e.g. .../data/)
            # - else: extract directly to .../afhq/
            if "afhq" in path_dir_target_x.parts:
                afhq_idx = path_dir_target_x.parts.index("afhq")
                path_dir_afhq = Path(*path_dir_target_x.parts[:afhq_idx + 1])
                path_dir_parent_afhq = path_dir_afhq.parent
            else:
                path_dir_parent_afhq = path_dir_target_x.parent.parent
                path_dir_afhq = path_dir_parent_afhq
            # end if

            if has_afhq_root:
                extract_destination = path_dir_parent_afhq
            else:
                extract_destination = path_dir_afhq
            # end if

            extract_destination.mkdir(parents=True, exist_ok=True)
            logger.info(f"Extracting files into destination: {extract_destination}")

            for member_item in tqdm(all_names, desc="Extracting"):
                zip_ref.extract(member_item, path=extract_destination)
            # end for
        # end with

        # If extraction placed files into a slightly different nested structure, handle symlink or move
        self.__verify_and_align_structure(extract_destination, path_dir_target_x, path_dir_target_y)
    # end def

    def __verify_and_align_structure(
        self,
        path_dir_extracted: Path,
        path_dir_target_x: Path,
        path_dir_target_y: Path,
    ) -> None:
        """Align directory structure if extracted root differs from target."""
        # If target X doesn't exist but train/cat exists under path_dir_extracted
        if not path_dir_target_x.exists():
            candidates = list(path_dir_extracted.rglob("cat"))
            for candidate in candidates:
                if candidate.is_dir() and candidate.parent.name == "train":
                    logger.info(f"Found candidate directory for X: {candidate}")
                    path_dir_target_x.parent.mkdir(parents=True, exist_ok=True)
                    if not path_dir_target_x.exists():
                        shutil.move(str(candidate), str(path_dir_target_x))
                    # end if
                    break
                # end if
            # end for
        # end if

        if not path_dir_target_y.exists():
            candidates = list(path_dir_extracted.rglob("dog"))
            for candidate in candidates:
                if candidate.is_dir() and candidate.parent.name == "train":
                    logger.info(f"Found candidate directory for Y: {candidate}")
                    path_dir_target_y.parent.mkdir(parents=True, exist_ok=True)
                    if not path_dir_target_y.exists():
                        shutil.move(str(candidate), str(path_dir_target_y))
                    # end if
                    break
                # end if
            # end for
        # end if
    # end def
# end class


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Download and set up the AFHQ cat and dog face dataset from config."
    )
    default_config_path = (
        Path(__file__).parent / "cat_and_dog_config" / "cat_and_dog_config.toml"
    )
    parser.add_argument(
        "--path_config",
        type=str,
        default=str(default_config_path),
        help=f"Path to the TOML configuration file (default: {default_config_path})",
    )
    parser.add_argument(
        "--url_dataset",
        type=str,
        default=DEFAULT_AFHQ_URL,
        help="URL of the dataset archive zip file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download and re-extraction even if dataset already exists.",
    )
    parser.add_argument(
        "--clean_zip",
        action="store_true",
        help="Remove the downloaded zip archive file after successful extraction.",
    )
    return parser
# end def


def main() -> None:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    path_config = Path(args.path_config)
    executor = DatasetSetupExecutor(
        path_file_config=path_config,
        url_dataset_archive=args.url_dataset,
        is_force_download=args.force,
        is_clean_zip=args.clean_zip,
    )

    try:
        result = executor.execute_setup_dataset()
        logger.info(f"Setup finished successfully: {result.is_success}")
    except Exception as exc:
        logger.error(f"Dataset setup failed: {exc}")
        sys.exit(1)
    # end try
# end def


if __name__ == "__main__":
    main()
# end if
