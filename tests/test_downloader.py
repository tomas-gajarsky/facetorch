import os
import pytest
from unittest.mock import patch

from facetorch.downloader import DownloaderGDrive, DownloaderHuggingFace


@pytest.mark.unit
@pytest.mark.downloader
class TestDownloaderGDrive:

    @patch("facetorch.downloader.gdown.download")
    @patch("facetorch.downloader.os.makedirs")
    def test_run(self, mock_makedirs, mock_download, tmp_path):
        dl = DownloaderGDrive(file_id="abc123", path_local=str(tmp_path / "model.pt"))
        dl.run()
        mock_makedirs.assert_called_once()
        mock_download.assert_called_once()


@pytest.mark.unit
@pytest.mark.downloader
class TestDownloaderHuggingFace:

    @patch("facetorch.downloader.hf_hub_download")
    def test_run_same_path(self, mock_hf_download, tmp_path):
        local = str(tmp_path / "model.pt")
        mock_hf_download.return_value = local
        dl = DownloaderHuggingFace(
            file_id="x", path_local=local, repo_id="user/repo", filename="model.pt"
        )
        dl.run()
        mock_hf_download.assert_called_once()

    @patch("facetorch.downloader.hf_hub_download")
    def test_run_different_path_renames(self, mock_hf_download, tmp_path):
        local = str(tmp_path / "model.pt")
        alt = str(tmp_path / "other.pt")
        with open(alt, "w") as f:
            f.write("data")
        mock_hf_download.return_value = alt
        dl = DownloaderHuggingFace(
            file_id="x", path_local=local, repo_id="user/repo", filename="model.pt"
        )
        dl.run()
        assert os.path.exists(local)
        assert not os.path.exists(alt)

    @patch("facetorch.downloader.hf_hub_download")
    def test_run_replaces_existing(self, mock_hf_download, tmp_path):
        local = str(tmp_path / "model.pt")
        alt = str(tmp_path / "other.pt")
        with open(local, "w") as f:
            f.write("old")
        with open(alt, "w") as f:
            f.write("new")
        mock_hf_download.return_value = alt
        dl = DownloaderHuggingFace(
            file_id="x", path_local=local, repo_id="user/repo", filename="model.pt"
        )
        dl.run()
        assert os.path.exists(local)

    @patch("facetorch.downloader.hf_hub_download", side_effect=Exception("network"))
    def test_run_failure_raises(self, mock_hf_download, tmp_path):
        local = str(tmp_path / "model.pt")
        dl = DownloaderHuggingFace(
            file_id="x", path_local=local, repo_id="user/repo", filename="model.pt"
        )
        with pytest.raises(Exception, match="network"):
            dl.run()

    def test_defaults(self):
        dl = DownloaderHuggingFace(file_id="user/repo", path_local="/tmp/model.pt")
        assert dl.repo_id == "user/repo"
        assert dl.filename == "model.pt"
