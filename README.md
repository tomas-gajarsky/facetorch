# ![](https://raw.githubusercontent.com/tomas-gajarsky/facetorch/main/data/facetorch-logo-42.png "facetorch logo") facetorch
![build](https://github.com/tomas-gajarsky/facetorch/actions/workflows/build.yml/badge.svg?branch=main)
![lint](https://github.com/tomas-gajarsky/facetorch/actions/workflows/lint.yml/badge.svg?branch=main)
[![PyPI](https://img.shields.io/pypi/v/facetorch)](https://pypi.org/project/facetorch/)
[![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/facetorch)](https://anaconda.org/conda-forge/facetorch)
[![PyPI - License](https://img.shields.io/pypi/l/facetorch)](https://raw.githubusercontent.com/tomas-gajarsky/facetorch/main/LICENSE)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

[Hugging Face Space demo app 🤗](https://huggingface.co/spaces/tomas-gajarsky/facetorch-app)

[Google Colab notebook demo](https://colab.research.google.com/github/tomas-gajarsky/facetorch/blob/main/notebooks/facetorch_notebook_demo.ipynb)
<a target="_blank" href="https://colab.research.google.com/github/tomas-gajarsky/facetorch/blob/main/notebooks/facetorch_notebook_demo.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

[User Guide](https://medium.com/@gajarsky.tomas/facetorch-user-guide-a0e9fd2a5552), [Documentation](https://tomas-gajarsky.github.io/facetorch/facetorch/index.html), [ChatGPT facetorch guide](https://chat.openai.com/g/g-q8HWAkG4u-facetorch-guide)
 
[Docker Hub](https://hub.docker.com/repository/docker/tomasgajarsky/facetorch) [(GPU)](https://hub.docker.com/repository/docker/tomasgajarsky/facetorch-gpu)


**Facetorch** is a Python library designed for facial detection and analysis, leveraging the power of deep neural networks. Its primary aim is to curate open-source face analysis models from the community, optimize them for high performance using TorchScript, and integrate them into a versatile face analysis toolkit. The library offers the following key features:

1. **Customizable Configuration:** Easily configure your setup using [Hydra](https://hydra.cc/docs/intro/) and its powerful [OmegaConf](https://omegaconf.readthedocs.io/) capabilities.

2. **Reproducible Environments:** Ensure reproducibility with tools like [conda-lock](https://github.com/conda-incubator/conda-lock) for dependency management and [Docker](https://docs.docker.com/get-docker/) for containerization.

3. **Accelerated Performance:** Enjoy enhanced performance on both CPU and GPU with [TorchScript](https://pytorch.org/docs/stable/jit.html) optimization.

4. **Simple Extensibility:** Extend the library by uploading your model file to Hugging Face Hub (previously Google Drive) and adding a corresponding configuration YAML file to the repository.

Facetorch provides an efficient, scalable, and user-friendly solution for facial analysis tasks, catering to developers and researchers looking for flexibility and performance.

Please use this library responsibly and with caution. Adhere to the [European Commission's Ethics Guidelines for Trustworthy AI](https://ec.europa.eu/futurium/en/ai-alliance-consultation.1.html) to ensure ethical and fair usage. Keep in mind that the models may have limitations and potential biases, so it is crucial to evaluate their outputs critically and consider their impact.


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
            ├── au
            ├── va
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

#### Facial Action Unit Detection (au)

    |        model        |   source  |  params |       license      | version |  
    | ------------------- | --------- | ------- | ------------------ | ------- |
    | OpenGraph Swin Base |  CVI-SZU  |   94M   |     MIT License    |    1    |

1. CVI-SZU
    * code: [ME-GraphAU](https://github.com/CVI-SZU/ME-GraphAU)
    * paper: [Luo et al. - Learning Multi-dimensional Edge Feature-based AU Relation Graph for Facial Action Unit Recognition](https://arxiv.org/abs/2205.01782)
    * [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-multi-dimensional-edge-feature-based/facial-action-unit-detection-on-bp4d)](https://paperswithcode.com/sota/facial-action-unit-detection-on-bp4d?p=learning-multi-dimensional-edge-feature-based)
    * ! Does not work with CUDA > 12.0

#### Facial Valence Arousal (va)

    |       model       |   source   |  params |   license   | version |
    | ----------------- | ---------- | ------- | ----------- | ------- |
    |  ELIM AL AlexNet  | kdhht2334  |  2.3M   | MIT license |    1    |

1. kdhht2334
    * code: [ELIM](https://github.com/kdhht2334/ELIM_FER)
    * paper: [Kim et al. - Optimal Transport-based Identity Matching
for Identity-invariant Facial Expression Recognition](https://arxiv.org/abs/2209.12172)

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

Models are downloaded during runtime automatically to the *models* directory using Hugging Face Hub (the default downloader has been switched from Google Drive to Hugging Face Hub).
Models are available on the [Hugging Face Hub](https://huggingface.co/tomas-gajarsky) and legacy models can also be accessed from the [original Google Drive folder](https://drive.google.com/drive/folders/19qlklR18wYfFsCChQ78it10XciuTzbDM?usp=sharing).


### Execution time

Image test.jpg (4 faces) is analyzed (including drawing boxes and landmarks, but not saving) in about 486ms and test3.jpg (25 faces) in about 1845ms (batch_size=8) on NVIDIA Tesla T4 GPU once the default configuration (*conf/config.yaml*) of models is initialized and pre heated to the initial image size 1080x1080 by the first run. One can monitor the execution times in logs using the DEBUG level.

Detailed test.jpg execution times:
```
analyzer
    ├── reader: 27 ms
    ├── detector: 193 ms
    ├── unifier: 1 ms
    └── predictor
            ├── embed: 8 ms
            ├── verify: 58 ms
            ├── fer: 28 ms
            ├── au: 57 ms
            ├── va: 1 ms
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
1. File of the TorchScript model
2. Repository on Hugging Face Hub for hosting the model (or legacy ID of the Google Drive model file)
3. facetorch [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo)

Facetorch works with models that were exported from PyTorch to TorchScript. You can apply [torch.jit.trace](https://pytorch.org/docs/stable/generated/torch.jit.trace.html) function to compile a PyTorch model as a TorchScript module. Please verify that the output of the traced model equals the output of the original model.

Models are now hosted on [Hugging Face Hub](https://huggingface.co/tomas-gajarsky) which is the default download source. You can host your model on your own Hugging Face account or use the legacy Google Drive hosting option by specifying the appropriate downloader in your configuration.

#### Configuration
##### Create yaml file
1. Create new folder with a short name of the task in predictor configuration directory 
```/conf/analyzer/predictor/``` following the FER example in ```/conf/analyzer/predictor/fer/```
2. Copy the yaml file ```/conf/analyzer/predictor/fer/efficientnet_b2_8.yaml``` to the new folder 
```/conf/analyzer/predictor/<predictor_name>/```
3. Change the yaml file name to the model you want to use: 
```/conf/analyzer/predictor/<predictor_name>/<model_name>.yaml```

##### Edit yaml file
1. Set up the downloader configuration:
   - For Hugging Face Hub (recommended): specify the `repo_id` and `filename` parameters
   - For legacy Google Drive: specify the Google Drive file ID
2. Select the preprocessor (or implement a new one based on BasePredPreProcessor) and specify its parameters e.g. image size and normalization in the yaml file 
to match the requirements of the new model.
3. Select the postprocessor (or implement a new one based on BasePredPostProcessor) and specify its parameters e.g. labels in the yaml file to match 
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

## Research Highlights Leveraging facetorch

### [Sharma et al. (2024)](https://aclanthology.org/2024.signlang-1.39.pdf)

Sharma, Paritosh, Camille Challant, and Michael Filhol. "Facial Expressions for Sign Language Synthesis using FACSHuman and AZee." *Proceedings of the LREC-COLING 2024 11th Workshop on the Representation and Processing of Sign Languages*, pp. 354–360, 2024.

### [Liang et al. (2023)](https://dl.acm.org/doi/abs/10.1145/3581783.3612854)

Liang, Cong, Jiahe Wang, Haofan Zhang, Bing Tang, Junshan Huang, Shangfei Wang, and Xiaoping Chen. "Unifarn: Unified transformer for facial reaction generation." *Proceedings of the 31st ACM International Conference on Multimedia*, pp. 9506–9510, 2023.

### [Gue et al. (2023)](https://research.monash.edu/en/publications/facial-expression-recognition-as-markers-of-depression)

Gue, Jia Xuan, Chun Yong Chong, and Mei Kuan Lim. "Facial Expression Recognition as markers of Depression." *2023 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)*, pp. 674–680, 2023.


## Acknowledgements
I would like to thank the open-source community and the researchers who have shared their work and published models. This project would not have been possible without their contributions.


## Citing

If you use facetorch in your work, please make sure to appropriately credit the original authors of the models it employs. Additionally, you may consider citing the facetorch library itself. Below is an example citation for facetorch:

```
@misc{facetorch,
    author = {Gajarsky, Tomas},
    title = {Facetorch: A Python Library for Analyzing Faces Using PyTorch},
    year = {2024},
    publisher = {GitHub},
    journal = {GitHub Repository},
    howpublished = {\url{https://github.com/tomas-gajarsky/facetorch}}
}
```
