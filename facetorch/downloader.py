import os
import re
from typing import Dict, List, Optional, Tuple

import gdown
from codetiming import Timer
from huggingface_hub import hf_hub_download

from facetorch import base
from facetorch.logger import LoggerJsonFile

logger = LoggerJsonFile().logger


class DownloaderGDrive(base.BaseDownloader):
    def __init__(self, file_id: str, path_local: str):
        """Downloader for Google Drive files.

        Args:
            file_id (str): ID of the file hosted on Google Drive.
            path_local (str): The file is downloaded to this local path.
        """
        super().__init__(file_id, path_local)

    @Timer("DownloaderGDrive.run", "{name}: {milliseconds:.2f} ms", logger=logger.debug)
    def run(self):
        """Downloads a file from Google Drive."""
        dir_local = os.path.dirname(self.path_local)
        if dir_local:
            os.makedirs(dir_local, exist_ok=True)
        url = f"https://drive.google.com/uc?&id={self.file_id}&confirm=t"
        gdown.download(url, output=self.path_local, quiet=False)


class DownloaderHuggingFace(base.BaseDownloader):
    def __init__(
        self,
        file_id: str,
        path_local: str,
        repo_id: str = None,
        filename: str = None,
        export_filenames_by_torch_minor: Optional[Dict[str, str]] = None,
        fallback_filenames: Optional[List[str]] = None,
        enable_default_torch_export_routing: bool = True,
    ):
        """Downloader for HuggingFace Hub files.

        This downloader retrieves model files from the HuggingFace Hub, serving as an alternative
        to Google Drive for storing and accessing facetorch models. This allows for better
        discoverability, versioning, and reliability compared to Google Drive links.

        Args:
            file_id (str): Not directly used for HuggingFace downloads, but kept for API compatibility.
                Can be used as a fallback for repo_id if repo_id is not provided.
            path_local (str): The file is downloaded to this local path.
            repo_id (str, optional): HuggingFace Hub repository ID in the format 'username/repo_name'.
                If not provided, attempts to parse from file_id.
            filename (str, optional): Name of the file to download from the repository.
                If not provided, uses the basename from path_local.
            export_filenames_by_torch_minor (Optional[Dict[str, str]]): Mapping from torch
                major.minor version (e.g. "2.11") to export filename (e.g.
                "model-torch2.11.pt2"). The downloader selects the best compatible
                filename for the current torch version before trying the generic
                filename, then falls back to older entries.
            fallback_filenames (Optional[List[str]]): Additional fallback filenames to try
                if the preferred filename cannot be downloaded.
            enable_default_torch_export_routing (bool): If True and filename ends with
                ".pt2", automatically enables fallback cohort names following
                ("*-torch2.3.pt2", "*-torch2.6.pt2", "*-torch2.11.pt2").
                If no explicit export_filenames_by_torch_minor map is supplied, the
                configured filename is still tried first.
        """
        super().__init__(file_id, path_local)
        self.repo_id = repo_id if repo_id else file_id
        self.filename = filename if filename else os.path.basename(path_local)
        self.export_filenames_by_torch_minor = export_filenames_by_torch_minor or {}
        self.fallback_filenames = fallback_filenames or []
        self.enable_default_torch_export_routing = enable_default_torch_export_routing

        self._active_candidate_index = -1
        self._active_filename = None
        self._last_candidates: List[str] = []

    @staticmethod
    def _parse_version_key(version_key: str) -> Optional[Tuple[int, int]]:
        """Parse a `major.minor` string into a comparable tuple."""
        match = re.fullmatch(r"\s*(\d+)\.(\d+)\s*", str(version_key))
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))

    @staticmethod
    def _current_torch_major_minor() -> Optional[Tuple[int, int]]:
        """Extract current torch major.minor from torch.__version__ safely."""
        try:
            import torch

            match = re.search(r"(\d+)\.(\d+)", str(torch.__version__))
            if not match:
                return None
            return int(match.group(1)), int(match.group(2))
        except Exception:
            return None

    def _default_export_filename_map(self) -> Dict[str, str]:
        """Build default export filename cohorts from the base filename."""
        if not self.filename.endswith(".pt2"):
            return {}

        stem, ext = os.path.splitext(self.filename)
        if "-torch" in stem:
            return {}

        return {
            "2.3": f"{stem}-torch2.3{ext}",
            "2.6": f"{stem}-torch2.6{ext}",
            "2.11": f"{stem}-torch2.11{ext}",
        }

    def _ordered_export_filenames(
        self, export_map: Dict[str, str], current_version: Optional[Tuple[int, int]]
    ) -> List[str]:
        """Order mapped export filenames by best compatibility with current torch."""
        parsed: List[Tuple[Tuple[int, int], str]] = []
        for key, filename in export_map.items():
            version_tuple = self._parse_version_key(key)
            if version_tuple is not None:
                parsed.append((version_tuple, filename))

        if not parsed:
            return []

        parsed.sort(key=lambda item: item[0])

        if current_version is None:
            return [filename for _, filename in reversed(parsed)]

        compatible = [item for item in parsed if item[0] <= current_version]
        incompatible_newer = [item for item in parsed if item[0] > current_version]

        ordered = list(reversed(compatible))
        ordered.extend(incompatible_newer)
        return [filename for _, filename in ordered]

    def _build_candidate_filenames(self) -> List[str]:
        """Build ordered list of candidate filenames to try on HF Hub."""
        current_version = self._current_torch_major_minor()

        explicit_export_map = dict(self.export_filenames_by_torch_minor)
        export_map = dict(explicit_export_map)
        if not explicit_export_map and self.enable_default_torch_export_routing:
            export_map = self._default_export_filename_map()

        ordered_exports = self._ordered_export_filenames(export_map, current_version)
        candidates: List[str] = []
        if explicit_export_map:
            candidates.extend(ordered_exports)
            candidates.append(self.filename)
        else:
            # Custom repos with only model.pt2 should not pay versioned 404s by default.
            candidates.append(self.filename)
            candidates.extend(ordered_exports)
        candidates.extend(self.fallback_filenames)

        # Final fallback for legacy TorchScript model in the same repo.
        if self.filename.endswith(".pt2"):
            stem, _ = os.path.splitext(self.filename)
            candidates.append(f"{stem}.pt")

        # De-duplicate while preserving order.
        return list(dict.fromkeys(candidates))

    def _download_one_candidate(self, filename: str, force_download: bool = False):
        """Download a single candidate filename from HF and place it at path_local."""
        local_dir = os.path.dirname(self.path_local) or "."
        downloaded_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=filename,
            local_dir=local_dir,
            force_download=force_download,
        )

        if os.path.abspath(downloaded_path) != os.path.abspath(self.path_local):
            if os.path.exists(self.path_local):
                os.remove(self.path_local)
            os.replace(downloaded_path, self.path_local)

    def _download_from_candidates(
        self,
        start_index: int = 0,
        force_download: bool = False,
        raise_on_failure: bool = True,
    ) -> bool:
        """Try candidate filenames in order; return True on first success."""
        candidates = self._build_candidate_filenames()
        self._last_candidates = candidates
        start_index = max(0, start_index)

        last_error = None
        for idx in range(start_index, len(candidates)):
            candidate = candidates[idx]
            try:
                self._download_one_candidate(candidate, force_download=force_download)
                self._active_candidate_index = idx
                self._active_filename = candidate
                logger.info(
                    f"Successfully downloaded {candidate} from {self.repo_id}"
                )
                return True
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Failed downloading {candidate} from {self.repo_id}: {e}"
                )

        if raise_on_failure:
            if last_error is not None:
                raise last_error
            raise RuntimeError(
                f"No download candidates available for repo {self.repo_id}."
            )
        return False

    @Timer(
        "DownloaderHuggingFace.run",
        "{name}: {milliseconds:.2f} ms",
        logger=logger.debug,
    )
    def run(self, force_download: bool = False):
        """Downloads a file from HuggingFace Hub.

        This method:
        1. Creates the necessary directory structure
        2. Resolves the best artifact filename for current torch version
        3. Downloads from HuggingFace Hub with fallback candidates
        4. Ensures the file is saved with the correct name at the specified path

        If the download fails, an informative error message is printed.
        """
        try:
            dir_local = os.path.dirname(self.path_local)
            if dir_local:
                os.makedirs(dir_local, exist_ok=True)
            self._download_from_candidates(
                start_index=0,
                force_download=force_download,
                raise_on_failure=True,
            )
        except Exception as e:
            logger.error(f"Error downloading from HuggingFace Hub: {e}")
            raise

    def try_next(self, force_download: bool = False) -> bool:
        """Try downloading the next candidate filename (used after load mismatch)."""
        return self._download_from_candidates(
            start_index=self._active_candidate_index + 1,
            force_download=force_download,
            raise_on_failure=False,
        )
