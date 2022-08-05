# ![](https://raw.githubusercontent.com/tomas-gajarsky/facetorch/main/data/facetorch-logo-42.png "Facetorch logo") facetorch 
[![PyPI](https://img.shields.io/pypi/v/facetorch)](https://pypi.org/project/facetorch/)
[![PyPI - License](https://img.shields.io/pypi/l/facetorch)](https://raw.githubusercontent.com/tomas-gajarsky/facetorch/main/LICENSE)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

[documentation](https://tomas-gajarsky.github.io/facetorch/facetorch/index.html)

Facetorch is a Python library that can detect faces and analyze facial features like expressions using artificial neural networks. The goal is to gather open-source face analysis models from the community, optimize them for performance using TorchScript and combine them to create a face analysis tool that one can:

1. configure using [Hydra](https://hydra.cc/docs/intro/) (OmegaConf)
2. reproduce with [conda-lock](https://github.com/conda-incubator/conda-lock) and [Docker](https://docs.docker.com/get-docker/)
3. accelerate on CPU and GPU with [TorchScript](https://pytorch.org/docs/stable/jit.html)
4. extend by uploading a model file to Google Drive and adding a config yaml file to the repository

Please, use the library responsibly with caution and follow the 
[ethics guidelines for Trustworthy AI from European Commission](https://ec.europa.eu/futurium/en/ai-alliance-consultation.1.html). 
The models are not perfect and may be biased.

## Install
PyPI
```bash
pip install facetorch
```

## Usage

### Prerequisites
* [Docker](https://docs.docker.com/get-docker/)
* [Docker Compose](https://docs.docker.com/compose/install/)

Docker Compose provides an easy way of building a working facetorch environment with a single command.

### Run docker example
    
* CPU: ```docker compose run facetorch python ./scripts/example.py```
* GPU: ```docker compose run facetorch-gpu python ./scripts/example.py analyzer.device=cuda```

### Configure

The project is configured by files located in *conf* with the main file *conf/config.yaml*.

## Components
FaceAnalyzer is the main class of Facetorch as it is the orchestrator responsible for initializing and running the following components:

1. Reader - reads the image and returns an ImageData object containing the image tensor.
2. Detector - wrapper around a neural network that detects faces.
3. Unifier - processor that unifies sizes of all faces and normalizes them
    between 0 and 1.
4. Predictor dict - set of wrappers around neural networks trained to analyze facial features.

### Structure
```
analyzer
    ├── reader
    ├── detector
    ├── unifier
    └── predictor
            ├── embed
            ├── fer
            └── deepfake
```


## Available models

### Detector

    |    detector   |   source  |   license   | version |
    | ------------- | --------- | ----------- | ------- |
    |   RetinaFace  |  biubug6  | MIT license |    1    |

1. biubug6
    * code: [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
    * paper: [RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild](https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.html)


### Predictor

#### Facial representation learning (embed)

    |       embed       |   source   |   license   | version |  
    | ----------------- | ---------- | ----------- | ------- |
    |  ResNet-50 VGG 1M |  1adrianb  | MIT license |    1    |

1. 1adrianb
    * code: [unsupervised-face-representation](https://github.com/1adrianb/unsupervised-face-representation)
    * paper: [Bulat et al. - Pre-training strategies and datasets for facial representation learning](https://arxiv.org/abs/2103.16554)


#### Facial expression recognition (FER)

    |        fer        |      source    |       license      | version |  
    | ----------------- | -------------- | ------------------ | ------- |
    | EfficientNet B0 7 | HSE-asavchenko | Apache License 2.0 |    1    |
    | EfficientNet B2 8 | HSE-asavchenko | Apache License 2.0 |    2    |

1. HSE-asavchenko
    * code: [face-emotion-recognition](https://github.com/HSE-asavchenko/face-emotion-recognition)
    * paper: [Savchenko - Facial expression and attributes recognition based on multi-task learning of lightweight neural networks](https://ieeexplore.ieee.org/abstract/document/9582508)

#### Deepfake detection

    |       deepfake       |      source      |   license   | version |
    | -------------------- | ---------------- | ----------- | ------- |
    |    EfficientNet B7   |     selimsef     | MIT license |    1    |

1. selimsef
    * code: [dfdc_deepfake_challenge](https://github.com/selimsef/dfdc_deepfake_challenge)
    * challenge: [Seferbekov - Deepfake Detection Challenge 1st place solution](https://www.kaggle.com/competitions/deepfake-detection-challenge/discussion)



### Model download

Models are downloaded during runtime automatically to the *models* directory.
You can also download the models manually from a [public Google Drive folder](https://drive.google.com/drive/folders/19qlklR18wYfFsCChQ78it10XciuTzbDM?usp=sharing).


### Execution time
Image test.jpg (4 faces) is analyzed in about 400ms and test3.jpg (25 faces) in about 1.1s on NVIDIA Tesla T4 GPU once the default configuration (*conf/config.yaml*) of models is initialized and pre heated to the initial image size 1080x1080. One can monitor the execution times in logs using the DEBUG level.


Detailed test.jpg (4 faces) execution times:
```
analyzer
    ├── reader: 27 ms
    ├── detector: 230 ms
    ├── unifier: 1 ms
    └── predictor
            ├── embed: 8 ms
            ├── fer: 22 ms
            └── deepfake: 109 ms
```


## Development
Run the Docker container:
* CPU: ```docker compose -f docker-compose.dev.yml run facetorch-dev bash```
* GPU: ```docker compose -f docker-compose.dev.yml run facetorch-dev-gpu bash```

### Add predictor
#### Prerequisites
1. File of the TorchScript model
2. Google Drive file ID of the model

Facetorch works with models that were exported from PyTorch to TorchScript. You can apply [torch.jit.trace](https://pytorch.org/docs/stable/generated/torch.jit.trace.html) function to compile a PyTorch model as a TorchScript module.

The first models are hosted on my [public Google Drive folder](https://drive.google.com/drive/folders/19qlklR18wYfFsCChQ78it10XciuTzbDM?usp=sharing). You can either send the new model for upload to me, host the model on your 
Google Drive or host it somewhere else and add your own downloader object to the codebase.

#### Configuration
##### Create yaml file
1. Create new folder with a short name of the task in predictor configuration directory 
```/conf/analyzer/predictor/``` following the FER example in ```/conf/analyzer/predictor/fer/```
2. Copy the yaml file ```/conf/analyzer/predictor/fer/efficientnet_b2_8.yaml``` to the new folder 
```/conf/analyzer/predictor/<predictor_name>/```
3. Change the yaml file name to the model you want to use: 
```/conf/analyzer/predictor/<predictor_name>/<model_name>.yaml```

##### Edit yaml file
1. Change the Google Drive file ID to the ID of the model.
2. Select the preprocessor (or implement a new one based on BasePredPreProcessor) and specify it's parameters e.g. image size and normalization in the yaml file 
to match the requirements of the new model.
3. Select the postprocessor (or implement a new one based on BasePredPostProcessor) and specify it's parameters e.g. labels in the yaml file to match 
the requirements of the new model.

##### Configure tests
1. Add a new predictor to the main *config.yaml* and all *tests.config.<n>.yaml* files. Alternatively, create a new config file e.g. 
*tests.config.<n>.yaml* and add it to the ```/tests/conftest.py``` file.
2. Write a test for the new predictor in ```/tests/test_<predictor_name>.py```

#### Test and submit
1. Run linting test: ```flake8 --config=.flake8```
2. Run tests and check coverage: ```pytest tests --verbose --cov-report html:coverage --cov facetorch```
3. Add the new predictor to the README model table.
4. Submit a pull request to add the new predictor to the main codebase.


### Update environment
* Add packages with corresponding versions to ```environment.yml``` file
* Lock the environment: ```conda lock -p linux-64 -f environment.yml```
* Install the locked environment: ```conda-lock install --name env conda-lock.yml```

### Generate documentation
* Generate documentation from docstrings using pdoc3:  ```pdoc --html facetorch --output-dir docs --force --template-dir pdoc/templates/```

### Profiling
1. Run profiling of the example script: ```python -m cProfile -o profiling/example.prof scripts/example.py```
2. Open profiling file in the browser: ```snakeviz profiling/example.prof```

## Acknowledgements
I want to thank the open source code community and the researchers who have published the models. This project would not be possible without their work.

![](https://raw.githubusercontent.com/tomas-gajarsky/facetorch/main/data/facetorch-logo-64.png "Facetorch logo")


Logo was generated using [DeepAI Text To Image API](https://deepai.org/machine-learning-model/text2img)