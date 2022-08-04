from typing import Optional, Union

import facetorch
import torch
from codetiming import Timer
from facetorch.analyzer.predictor.core import FacePredictor
from facetorch.datastruct import ImageData, Response
from facetorch.logger import LoggerJsonFile
from facetorch.utils import draw_boxes_and_save
from hydra.utils import instantiate
from omegaconf import OmegaConf

logger = LoggerJsonFile().logger


class FaceAnalyzer(object):
    @Timer("FaceAnalyzer.__init__", "{name}: {milliseconds:.2f} ms", logger.debug)
    def __init__(self, cfg: OmegaConf):
        """FaceAnalyzer is the main class that reads images and runs face detection, tensor unification, as well as facial feature analysis prediction.
        It is the orchestrator responsible for initializing and running the following components:

        1. Reader - reads the image and returns an ImageData object containing the image tensor.
        2. Detector - wrapper around a neural network that detects faces.
        3. Unifier - processor that unifies sizes of all faces and normalizes them between 0 and 1.
        4. Predictor list - list of wrappers around models trained to analyze facial features for example, expressions.

        Args:
            cfg (OmegaConf): Config object with image reader, face detector, unifier and predictor configurations.

        Attributes:
            cfg (OmegaConf): Config object with image reader, face detector, unifier and predictor configurations.
            reader (BaseReader): Reader object that reads the image and returns an ImageData object containing the image tensor.
            detector (FaceDetector): FaceDetector object that wraps a neural network that detects faces.
            unifier (FaceUnifier): FaceUnifier object that unifies sizes of all faces and normalizes them between 0 and 1.
            predictors (Dict[str, FacePredictor]): Dict of FacePredictor objects that predict facial features. Key is the name of the predictor.
            logger (logging.Logger): Logger object that logs messages.

        """
        self.cfg = cfg
        self.logger = instantiate(self.cfg.logger).logger

        self.logger.info("Initializing FaceAnalyzer")
        self.logger.debug("Config", extra=self.cfg.__dict__["_content"])

        self.logger.info("Initializing BaseReader")
        self.reader = instantiate(self.cfg.reader)

        self.logger.info("Initializing FaceDetector")
        self.detector = instantiate(self.cfg.detector)

        self.logger.info("Initializing FaceUnifier")
        self.unifier = instantiate(self.cfg.unifier)

        self.logger.info("Initializing dictionary of FacePredictor objects")
        self.predictors = {}
        for predictor_name in self.cfg.predictor:
            self.logger.info(f"Initializing FacePredictor {predictor_name}")
            self.predictors[predictor_name] = instantiate(
                self.cfg.predictor[predictor_name]
            )

    @Timer("FaceAnalyzer.run", "{name}: {milliseconds:.2f} ms", logger.debug)
    def run(
        self,
        path_image: str,
        batch_size: int = 8,
        fix_img_size: bool = False,
        return_img_data: bool = False,
        include_tensors: bool = False,
        path_output: Optional[str] = None,
    ) -> Union[Response, ImageData]:
        """Reads image, detects faces, unifies the detected faces, predicts facial features
         and returns analyzed data.

        Args:
            path_image (str): Path to the input image.
            batch_size (int): Batch size for making predictions on the faces. Default is 8.
            fix_img_size (bool): If True, resizes the image to the size specified in reader. Default is False.
            return_img_data (bool): If True, returns all image data including tensors, otherwise only returns the faces. Default is False.
            include_tensors (bool): If True, removes tensors from the returned data object. Default is False.
            path_output (Optional[str]): Path where to save the image with detected faces. If None, the image is not saved. Default: None.

        Returns:
            Union[Response, ImageData]: If return_img_data is False, returns a Response object containing the faces and their facial features. If return_img_data is True, returns the entire ImageData object.

        """

        def _predict_batch(
            data: ImageData, predictor: FacePredictor, predictor_name: str
        ) -> ImageData:
            n_faces = len(data.faces)

            for face_indx_start in range(0, n_faces, batch_size):
                face_indx_end = min(face_indx_start + batch_size, n_faces)

                face_batch_tensor = torch.stack(
                    [face.tensor for face in data.faces[face_indx_start:face_indx_end]]
                )
                preds = predictor.run(face_batch_tensor)
                data.add_preds(preds, predictor_name, face_indx_start)

            return data

        self.logger.info("Running FaceAnalyzer")
        self.logger.info("Reading image", extra={"path_image": path_image})
        data = self.reader.run(path_image, fix_img_size=fix_img_size)
        data.version = facetorch.__version__

        self.logger.info("Detecting faces")
        data = self.detector.run(data)
        n_faces = len(data.faces)
        self.logger.info(f"Number of faces: {n_faces}")

        if n_faces > 0:
            self.logger.info("Unifying faces")
            data = self.unifier.run(data)

            self.logger.info("Predicting facial features")
            for predictor_name, predictor in self.predictors.items():
                self.logger.info(f"Running FacePredictor: {predictor_name}")
                data = _predict_batch(data, predictor, predictor_name)

        path_output = None if path_output == "None" else path_output
        if path_output is not None:
            self.logger.info("Saving image", extra={"path_output": path_output})
            draw_boxes_and_save(data, path_output)

        if not include_tensors:
            self.logger.info("Removing tensors")
            data.reset_tensors()

        response = Response(faces=data.faces, version=data.version)

        if return_img_data:
            self.logger.debug("Returning image data object", extra=data.__dict__)
            return data
        else:
            self.logger.debug("Returning response with faces", extra=response.__dict__)
            return response