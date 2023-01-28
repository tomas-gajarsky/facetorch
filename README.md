# ![](https://raw.githubusercontent.com/tomas-gajarsky/facetorch/main/data/facetorch-logo-42.png "facetorch logo") facetorch
![build](https://github.com/tomas-gajarsky/facetorch/actions/workflows/build.yml/badge.svg?branch=main)
![lint](https://github.com/tomas-gajarsky/facetorch/actions/workflows/lint.yml/badge.svg?branch=main)
[![PyPI](https://img.shields.io/pypi/v/facetorch)](https://pypi.org/project/facetorch/)
[![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/facetorch)](https://anaconda.org/conda-forge/facetorch)
[![PyPI - License](https://img.shields.io/pypi/l/facetorch)](https://raw.githubusercontent.com/tomas-gajarsky/facetorch/main/LICENSE)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

[Demo App on Hugging Face Spaces 🤗 ](https://huggingface.co/spaces/tomas-gajarsky/facetorch-app)

[User Guide](https://medium.com/@gajarsky.tomas/facetorch-user-guide-a0e9fd2a5552), [Documentation](https://tomas-gajarsky.github.io/facetorch/facetorch/index.html)
 
[Docker Hub](https://hub.docker.com/repository/docker/tomasgajarsky/facetorch) [(GPU)](https://hub.docker.com/repository/docker/tomasgajarsky/facetorch-gpu)

Facetorch is a Python library that can detect faces and analyze facial features using deep neural networks. The goal is to gather open sourced face analysis models from the community, optimize them for performance using TorchScript and combine them to create a face analysis tool that one can:

1. configure using [Hydra](https://hydra.cc/docs/intro/) (OmegaConf)
2. reproduce with [conda-lock](https://github.com/conda-incubator/conda-lock) and [Docker](https://docs.docker.com/get-docker/)
3. accelerate on CPU and GPU with [TorchScript](https://pytorch.org/docs/stable/jit.html)
4. extend by uploading a model file to Google Drive and adding a config yaml file to the repository

Please, use the library responsibly with caution and follow the 
[ethics guidelines for Trustworthy AI from European Commission](https://ec.europa.eu/futurium/en/ai-alliance-consultation.1.html). 
The models are not perfect and may be biased.

## Install
[PyPI](https://pypi.org/project/facetorch/)
```bash
pip install facetorch
```
[Conda](https://anaconda.org/conda-forge/facetorch)
```bash
conda install -c conda-forge facetorch
```
## Usage

### Prerequisites
* [Docker](https://docs.docker.com/get-docker/)
* [Docker Compose](https://docs.docker.com/compose/install/)

Docker Compose provides an easy way of building a working facetorch environment with a single command.

### Run docker example
    
* CPU: ```docker compose run facetorch python ./scripts/example.py```
* GPU: ```docker compose run facetorch-gpu python ./scripts/example.py analyzer.device=cuda```

Check *data/output* for resulting images with bounding boxes and facial 3D landmarks.

(Apple Mac M1) Use Rosetta 2 emulator in Docker Desktop to run the CPU version.

### Configure

The project is configured by files located in *conf* with the main file: *conf/config.yaml*. One can easily add or remove modules from the configuration.

## Components
FaceAnalyzer is the main class of facetorch as it is the orchestrator responsible for initializing and running the following components:

1. Reader - reads the image and returns an ImageData object containing the image tensor.
2. Detector - wrapper around a neural network that detects faces.
3. Unifier - processor that unifies sizes of all faces and normalizes them
    between 0 and 1.
4. Predictor dict - set of wrappers around neural networks trained to analyze facial features.
5. Utilizer dict - set of wrappers around any functionality that requires the output of neural networks e.g. drawing bounding boxes or facial landmarks.

### Structure
```
analyzer
    ├── reader
    ├── detector
    ├── unifier
    └── predictor
            ├── embed
            ├── verify
            ├── fer
            ├── deepfake
            └── align
    └── utilizer
            ├── align
            ├── draw
            └── save
```


## Models

### Detector

    |     model     |   source  |   params  |   license   | version |
    | ------------- | --------- | --------- | ----------- | ------- |
    |   RetinaFace  |  biubug6  |   27.3M   | MIT license |    1    |

1. biubug6
    * code: [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
    * paper: [Deng et al. - RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild](https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.html)
    * [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/190500641/face-detection-on-wider-face-hard)](https://paperswithcode.com/sota/face-detection-on-wider-face-hard?p=190500641)



### Predictor

#### Facial Representation Learning (embed)

    |       model       |   source   |  params |   license   | version |  
    | ----------------- | ---------- | ------- | ----------- | ------- |
    |  ResNet-50 VGG 1M |  1adrianb  |  28.4M  | MIT license |    1    |

1. 1adrianb
    * code: [unsupervised-face-representation](https://github.com/1adrianb/unsupervised-face-representation)
    * paper: [Bulat et al. - Pre-training strategies and datasets for facial representation learning](https://arxiv.org/abs/2103.16554)
    * Note: ```include_tensors``` needs to be True in order to include the model prediction in Prediction.logits

#### Face Verification (verify)

    |       model      |   source    |  params  |      license       | version |  
    | ---------------- | ----------- | -------- | ------------------ | ------- |
    |    MagFace+UNPG  | Jung-Jun-Uk |   65.2M  | Apache License 2.0 |    1    |
    |  AdaFaceR100W12M |  mk-minchul |    -     |     MIT License    |    2    |

1. Jung-Jun-Uk
    * code: [UNPG](https://github.com/jung-jun-uk/unpg)
    * paper: [Jung et al. - Unified Negative Pair Generation toward Well-discriminative Feature Space for Face Recognition](https://arxiv.org/abs/2203.11593)
    * [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unified-negative-pair-generation-toward-well/face-verification-on-ijb-b)](https://paperswithcode.com/sota/face-verification-on-ijb-b?p=unified-negative-pair-generation-toward-well)(FAR=0.01)
    * Note: ```include_tensors``` needs to be True in order to include the model prediction in Prediction.logits
2. mk-minchul
    * code: [AdaFace](https://github.com/mk-minchul/adaface)
    * paper: [Kim et al. - AdaFace: Quality Adaptive Margin for Face Recognition](https://arxiv.org/abs/2204.00964)
    * [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/adaface-quality-adaptive-margin-for-face/face-verification-on-ijb-b)](https://paperswithcode.com/sota/face-verification-on-ijb-b?p=adaface-quality-adaptive-margin-for-face) <
    * [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/adaface-quality-adaptive-margin-for-face/face-verification-on-ijb-c)](https://paperswithcode.com/sota/face-verification-on-ijb-c?p=adaface-quality-adaptive-margin-for-face) <
    * < badges represent models trained on smaller WebFace 4M dataset
    * Note: ```include_tensors``` needs to be True in order to include the model prediction in Prediction.logits


#### Facial Expression Recognition (fer)

    |       model       |      source    |  params  |       license      | version |  
    | ----------------- | -------------- | -------- | ------------------ | ------- |
    | EfficientNet B0 7 | HSE-asavchenko |    4M    | Apache License 2.0 |    1    |
    | EfficientNet B2 8 | HSE-asavchenko |   7.7M   | Apache License 2.0 |    2    |

1. HSE-asavchenko
    * code: [face-emotion-recognition](https://github.com/HSE-asavchenko/face-emotion-recognition)
    * paper: [Savchenko - Facial expression and attributes recognition based on multi-task learning of lightweight neural networks](https://ieeexplore.ieee.org/abstract/document/9582508)
    * B2 [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/classifying-emotions-and-engagement-in-online/facial-expression-recognition-on-affectnet)](https://paperswithcode.com/sota/facial-expression-recognition-on-affectnet?p=classifying-emotions-and-engagement-in-online)
    * B0 [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/facial-expression-and-attributes-recognition/facial-expression-recognition-on-affectnet)](https://paperswithcode.com/sota/facial-expression-recognition-on-affectnet?p=facial-expression-and-attributes-recognition)
    * B0 [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/facial-expression-and-attributes-recognition/facial-expression-recognition-on-acted-facial)](https://paperswithcode.com/sota/facial-expression-recognition-on-acted-facial?p=facial-expression-and-attributes-recognition)


#### Deepfake Detection (deepfake)

    |         model        |      source      |  params  |   license   | version |
    | -------------------- | ---------------- | -------- | ----------- | ------- |
    |    EfficientNet B7   |     selimsef     |   66.4M  | MIT license |    1    |

1. selimsef
    * code: [dfdc_deepfake_challenge](https://github.com/selimsef/dfdc_deepfake_challenge)
    * challenge: [Seferbekov - Deepfake Detection Challenge 1st place solution](https://www.kaggle.com/competitions/deepfake-detection-challenge/discussion)

#### Face Alignment (align)

    |       model       |      source      |  params  |   license   | version |
    | ----------------- | ---------------- | -------- | ----------- | ------- |
    |    MobileNet v2   |     choyingw     |   4.1M   | MIT license |    1    |

1. choyingw
    * code: [SynergyNet](https://github.com/choyingw/SynergyNet)
    * challenge: [Wu et al. - Synergy between 3DMM and 3D Landmarks for Accurate 3D Facial Geometry](https://arxiv.org/abs/2110.09772)
    * [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/synergy-between-3dmm-and-3d-landmarks-for/face-alignment-on-aflw)](https://paperswithcode.com/sota/face-alignment-on-aflw?p=synergy-between-3dmm-and-3d-landmarks-for)
    * [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/synergy-between-3dmm-and-3d-landmarks-for/head-pose-estimation-on-aflw2000)](https://paperswithcode.com/sota/head-pose-estimation-on-aflw2000?p=synergy-between-3dmm-and-3d-landmarks-for)
    * [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/synergy-between-3dmm-and-3d-landmarks-for/face-alignment-on-aflw2000-3d)](https://paperswithcode.com/sota/face-alignment-on-aflw2000-3d?p=synergy-between-3dmm-and-3d-landmarks-for)
    * Note: ```include_tensors``` needs to be True in order to include the model prediction in Prediction.logits



### Model download

Models are downloaded during runtime automatically to the *models* directory.
You can also download the models manually from a [public Google Drive folder](https://drive.google.com/drive/folders/19qlklR18wYfFsCChQ78it10XciuTzbDM?usp=sharing).


### Execution time

Image test.jpg (4 faces) is analyzed (including drawing boxes and landmarks, but not saving) in about 465ms and test3.jpg (25 faces) in about 1480ms (batch_size=8) on NVIDIA Tesla T4 GPU once the default configuration (*conf/config.yaml*) of models is initialized and pre heated to the initial image size 1080x1080 by the first run. One can monitor the execution times in logs using the DEBUG level.

Detailed test.jpg execution times:
```
analyzer
    ├── reader: 27 ms
    ├── detector: 230 ms
    ├── unifier: 1 ms
    └── predictor
            ├── embed: 8 ms
            ├── verify: 58 ms
            ├── fer: 28 ms
            ├── deepfake: 117 ms
            └── align: 5 ms
    └── utilizer
            ├── align: 8 ms
            ├── draw_boxes: 22 ms
            ├── draw_landmarks: 7 ms
            └── save: 298 ms
```


## Development
Run the Docker container:
* CPU: ```docker compose -f docker-compose.dev.yml run facetorch-dev```
* GPU: ```docker compose -f docker-compose.dev.yml run facetorch-dev-gpu```

### Add predictor
#### Prerequisites
1. file of the TorchScript model
2. ID of the Google Drive model file
3. facetorch [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo)

Facetorch works with models that were exported from PyTorch to TorchScript. You can apply [torch.jit.trace](https://pytorch.org/docs/stable/generated/torch.jit.trace.html) function to compile a PyTorch model as a TorchScript module. Please verify that the output of the traced model equals the output of the original model.

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
4. (Optional) Add BaseUtilizer derivative that uses output of your model to perform some additional actions.

##### Configure tests
1. Add a new predictor to the main *config.yaml* and all *tests.config.n.yaml* files. Alternatively, create a new config file e.g. 
*tests.config.n.yaml* and add it to the ```/tests/conftest.py``` file.
2. Write a test for the new predictor in ```/tests/test_<predictor_name>.py```

#### Test and submit
1. Run linting: ```black facetorch```
2. Add the new predictor to the README model table.
3. Update CHANGELOG and version
4. Submit a pull request to the repository


### Update environment
CPU:
* Add packages with corresponding versions to ```environment.yml``` file
* Lock the environment: ```conda lock -p linux-64 -f environment.yml --lockfile conda-lock.yml```
* (Alternative Docker) Lock the environment: ```docker compose -f docker-compose.dev.yml run facetorch-lock```
* Install the locked environment: ```conda-lock install --name env conda-lock.yml```

GPU:
* Add packages with corresponding versions to ```gpu.environment.yml``` file
* Lock the environment: ```conda lock -p linux-64 -f gpu.environment.yml --lockfile gpu.conda-lock.yml```
* (Alternative Docker) Lock the environment: ```docker compose -f docker-compose.dev.yml run facetorch-lock-gpu```
* Install the locked environment: ```conda-lock install --name env gpu.conda-lock.yml```

### Run tests + coverage
* Run tests and generate coverage: ```pytest tests --verbose --cov-report html:coverage --cov facetorch```

### Generate documentation
* Generate documentation from docstrings using pdoc3:  ```pdoc --html facetorch --output-dir docs --force --template-dir pdoc/templates/```

### Profiling
1. Run profiling of the example script: ```python -m cProfile -o profiling/example.prof scripts/example.py```
2. Open profiling file in the browser: ```snakeviz profiling/example.prof```

## Acknowledgements
I want to thank the open source code community and the researchers who have published the models. This project would not be possible without their work.

![](https://raw.githubusercontent.com/tomas-gajarsky/facetorch/main/data/facetorch-logo-64.png "facetorch logo")


Logo was generated using [DeepAI Text To Image API](https://deepai.org/machine-learning-model/text2img)