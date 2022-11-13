import os
import gdown
from codetiming import Timer

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
        os.makedirs(os.path.dirname(self.path_local), exist_ok=True)
        url = f"https://drive.google.com/uc?export=download&confirm=pbef&id={self.file_id}"
        gdown.download(url, output=self.path_local, quiet=False)
