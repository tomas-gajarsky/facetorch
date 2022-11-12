# Change Log


## 0.1.3

Released on November 12, 2022.

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
