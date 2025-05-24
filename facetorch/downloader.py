import os
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
        os.makedirs(os.path.dirname(self.path_local), exist_ok=True)
        url = f"https://drive.google.com/uc?&id={self.file_id}&confirm=t"
        gdown.download(url, output=self.path_local, quiet=False)


class DownloaderHuggingFace(base.BaseDownloader):
    def __init__(self, file_id: str, path_local: str, repo_id: str = None, filename: str = None):
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
        """
        super().__init__(file_id, path_local)
        self.repo_id = repo_id if repo_id else file_id
        self.filename = filename if filename else os.path.basename(path_local)

    @Timer("DownloaderHuggingFace.run", "{name}: {milliseconds:.2f} ms", logger=logger.debug)
    def run(self):
        """Downloads a file from HuggingFace Hub.
        
        This method:
        1. Creates the necessary directory structure
        2. Downloads the specified file from HuggingFace Hub
        3. Ensures the file is saved with the correct name at the specified path
        
        If the download fails, an informative error message is printed.
        """
        try:
            os.makedirs(os.path.dirname(self.path_local), exist_ok=True)
            
            # Download the file from HuggingFace Hub
            downloaded_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.filename,
                local_dir=os.path.dirname(self.path_local)
            )
            
            # Ensure the file is at the exact path specified in path_local
            if downloaded_path != self.path_local:
                # If HF Hub downloaded to a different path, move/rename it
                if os.path.exists(self.path_local):
                    os.remove(self.path_local)
                os.rename(downloaded_path, self.path_local)
                
            logger.info(f"Successfully downloaded {self.filename} from {self.repo_id}")
        except Exception as e:
            logger.error(f"Error downloading from HuggingFace Hub: {e}")
            raise
