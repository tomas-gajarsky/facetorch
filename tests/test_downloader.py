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

    @patch("facetorch.downloader.gdown.download")
    @patch("facetorch.downloader.os.makedirs")
    def test_run_basename_path(self, mock_makedirs, mock_download):
        dl = DownloaderGDrive(file_id="abc123", path_local="model.pt")
        dl.run()
        mock_makedirs.assert_not_called()
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
    @patch("facetorch.downloader.os.makedirs")
    def test_run_basename_path(
        self, mock_makedirs, mock_hf_download, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        downloaded = tmp_path / "downloaded.pt"
        downloaded.write_bytes(b"data")
        mock_hf_download.return_value = str(downloaded)
        dl = DownloaderHuggingFace(
            file_id="x", path_local="model.pt", repo_id="user/repo", filename="model.pt"
        )
        dl.run()
        mock_makedirs.assert_not_called()
        assert mock_hf_download.call_args.kwargs["local_dir"] == "."
        assert (tmp_path / "model.pt").exists()

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

    def test_default_pt2_candidate_order_for_newer_torch(self, tmp_path):
        dl = DownloaderHuggingFace(
            file_id="x",
            path_local=str(tmp_path / "model.pt2"),
            repo_id="user/repo",
            filename="model.pt2",
        )
        with patch.object(
            DownloaderHuggingFace, "_current_torch_major_minor", return_value=(2, 11)
        ):
            candidates = dl._build_candidate_filenames()

        assert candidates == [
            "model.pt2",
            "model-torch2.11.pt2",
            "model-torch2.6.pt2",
            "model-torch2.3.pt2",
            "model.pt",
        ]

    def test_custom_export_map_candidate_order(self, tmp_path):
        dl = DownloaderHuggingFace(
            file_id="x",
            path_local=str(tmp_path / "model.pt2"),
            repo_id="user/repo",
            filename="model.pt2",
            export_filenames_by_torch_minor={
                "2.3": "model-torch2.3.pt2",
                "2.11": "model-torch2.11.pt2",
            },
        )
        with patch.object(
            DownloaderHuggingFace, "_current_torch_major_minor", return_value=(2, 6)
        ):
            candidates = dl._build_candidate_filenames()

        assert candidates == [
            "model-torch2.3.pt2",
            "model-torch2.11.pt2",
            "model.pt2",
            "model.pt",
        ]

    def test_explicit_export_map_prefers_best_cohort(self, tmp_path):
        dl = DownloaderHuggingFace(
            file_id="x",
            path_local=str(tmp_path / "model.pt2"),
            repo_id="user/repo",
            filename="model.pt2",
            export_filenames_by_torch_minor={
                "2.3": "model-torch2.3.pt2",
                "2.6": "model-torch2.6.pt2",
                "2.11": "model-torch2.11.pt2",
            },
        )
        with patch.object(
            DownloaderHuggingFace, "_current_torch_major_minor", return_value=(2, 11)
        ):
            candidates = dl._build_candidate_filenames()

        assert candidates == [
            "model-torch2.11.pt2",
            "model-torch2.6.pt2",
            "model-torch2.3.pt2",
            "model.pt2",
            "model.pt",
        ]

    @patch("facetorch.downloader.hf_hub_download")
    def test_run_falls_back_between_candidates(self, mock_hf_download, tmp_path):
        local = str(tmp_path / "model.pt2")

        def _side_effect(repo_id, filename, local_dir, force_download=False):
            if filename == "model-torch2.11.pt2":
                raise Exception("missing")
            downloaded = str(tmp_path / filename)
            with open(downloaded, "wb") as f:
                f.write(b"ok")
            return downloaded

        mock_hf_download.side_effect = _side_effect
        dl = DownloaderHuggingFace(
            file_id="x",
            path_local=local,
            repo_id="user/repo",
            filename="model.pt2",
            export_filenames_by_torch_minor={
                "2.11": "model-torch2.11.pt2",
                "2.6": "model-torch2.6.pt2",
            },
        )
        with patch.object(
            DownloaderHuggingFace, "_current_torch_major_minor", return_value=(2, 11)
        ):
            dl.run()

        assert os.path.exists(local)
        assert dl._active_filename == "model-torch2.6.pt2"
        assert mock_hf_download.call_count == 2

    @patch("facetorch.downloader.hf_hub_download")
    def test_try_next_downloads_next_candidate(self, mock_hf_download, tmp_path):
        local = str(tmp_path / "model.pt2")
        seen_filenames = []

        def _side_effect(repo_id, filename, local_dir, force_download=False):
            seen_filenames.append((filename, force_download))
            downloaded = str(tmp_path / filename)
            with open(downloaded, "wb") as f:
                f.write(b"ok")
            return downloaded

        mock_hf_download.side_effect = _side_effect
        dl = DownloaderHuggingFace(
            file_id="x",
            path_local=local,
            repo_id="user/repo",
            filename="model.pt2",
        )
        with patch.object(
            DownloaderHuggingFace, "_current_torch_major_minor", return_value=(2, 11)
        ):
            dl.run()
            assert dl.try_next(force_download=True) is True

        assert seen_filenames[0][0] == "model.pt2"
        assert seen_filenames[1][0] == "model-torch2.11.pt2"
        assert seen_filenames[1][1] is True
