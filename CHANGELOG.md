# Change Log


## 0.3.0

Released on February 9, 2023.

### Added
* predictor for facial action unit detection - OpenGraphAU from Luo et al.

### Changed
* facetorch version retrieval error handling to not fail the run


## 0.2.4

Released on February 4, 2023.

### Changed
* scope of some test fixtures from function to session


## 0.2.3

Released on February 2, 2023.

### Fixed
* error in detector test

### Removed
* deletion of detector model in tests


## 0.2.2

Released on January 28, 2023.

### Fixed
* error in Google Colab demo notebook


## 0.2.1

Released on January 28, 2023.

### Added
* Google Colab demo notebook
* Google Colab demo notebook link to README
* Merged config yaml files for OmegaConf to /conf/merged/ directory

### Changed
* Do not fail the run if facetorch version cannot be retrieved (likely due to local installation)


## 0.2.0

Released on January 28, 2023.

### Added
* Python 3.8 support
* platform to Docker compose file for Apple Silicon M chips
* Docker compose service for locking dependencies


## 0.1.5

Released on January 22, 2023.

### Added
* number of parameters for each model in README

### Changed
* facetorch installation in dev Dockerfiles to not install dependencies from PyPI

### Removed
* unused port bindings from dev Docker compose file


## 0.1.4

Released on November 18, 2022.

### Added
* predictor for face verification and face recognition - AdaFace by Kim et al.


## 0.1.3

Released on November 13, 2022.

### Added
* Torchvision based landmark drawer as a utilizer

### Changed
* drawing of 3D landmarks is 21x faster: 152 ms -> 7 ms

### Removed
* Matplotlib based landmark drawer utilizer
* Matplotlib dependency


## 0.1.2

Released on November 10, 2022.

### Added
* links to Demo App and User Guide in README.md

### Changed
* Box expansion method changed from static absolute value to dynamic relative value


## 0.1.1

Released on August 31, 2022.

### Added
* predictor for face verification task - MagFace+UNPG by Jung et al.


## 0.1.0

Released on August 22, 2022.

### Added
* badges for models with SOTA comparison using papers with code
* predictor for face alignment task - SynergyNet
* utilizers for face alignment (compute 3d landmarks, mesh and pose) and drawing landmarks

### Changed
* default logging level to INFO
* drawing boxes and saving image abstracted to utilizer objects


## 0.0.8

Released on August 12, 2022.

### Added

* conda-forge documentation
* entrypoints to docker compose services

### Changed

* workflow for testing conda installation
* workflow for pushing facetorch-gpu Docker image to Docker Hub


## 0.0.7

Released on August 11, 2022.

### Added

* facetorch GitHub repository link to docs
* secrets to workflows that push to PyPI and Docker Hub


## 0.0.6

Released on August 10, 2022.

### Added

* GitHub workflows


## 0.0.5

Released on August 7, 2022.

### Added

* GPU specific conda environment


## 0.0.4

Released on August 6, 2022.

### Added

* Manifest file for distribution


## 0.0.3

Released on August 5, 2022.

### Added

* Logo


## 0.0.1

Released on August 5, 2022.

### Added

* First version of facetorch package, containing:
	- Tests,
	- Documentation,
	- Code style checking
	- Contributing guidelines
