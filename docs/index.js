URLS=[
"facetorch/index.html",
"facetorch/analyzer/index.html",
"facetorch/analyzer/core.html",
"facetorch/analyzer/detector/index.html",
"facetorch/base.html",
"facetorch/analyzer/detector/core.html",
"facetorch/analyzer/detector/post.html",
"facetorch/analyzer/detector/pre.html",
"facetorch/analyzer/predictor/index.html",
"facetorch/analyzer/predictor/core.html",
"facetorch/analyzer/predictor/post.html",
"facetorch/analyzer/predictor/pre.html",
"facetorch/analyzer/reader/index.html",
"facetorch/analyzer/reader/core.html",
"facetorch/analyzer/unifier/index.html",
"facetorch/analyzer/unifier/core.html",
"facetorch/datastruct.html",
"facetorch/downloader.html",
"facetorch/logger.html",
"facetorch/transforms.html",
"facetorch/utils.html"
];
INDEX=[
{
"ref":"facetorch",
"url":0,
"doc":""
},
{
"ref":"facetorch.FaceAnalyzer",
"url":0,
"doc":"FaceAnalyzer is the main class that reads images and runs face detection, tensor unification, as well as facial feature analysis prediction. It is the orchestrator responsible for initializing and running the following components: 1. Reader - reads the image and returns an ImageData object containing the image tensor. 2. Detector - wrapper around a neural network that detects faces. 3. Unifier - processor that unifies sizes of all faces and normalizes them between 0 and 1. 4. Predictor list - list of wrappers around models trained to analyze facial features for example, expressions. Args: cfg (OmegaConf): Config object with image reader, face detector, unifier and predictor configurations. Attributes: cfg (OmegaConf): Config object with image reader, face detector, unifier and predictor configurations. reader (BaseReader): Reader object that reads the image and returns an ImageData object containing the image tensor. detector (FaceDetector): FaceDetector object that wraps a neural network that detects faces. unifier (FaceUnifier): FaceUnifier object that unifies sizes of all faces and normalizes them between 0 and 1. predictors (Dict[str, FacePredictor]): Dict of FacePredictor objects that predict facial features. Key is the name of the predictor. logger (logging.Logger): Logger object that logs messages."
},
{
"ref":"facetorch.FaceAnalyzer.run",
"url":0,
"doc":"Reads image, detects faces, unifies the detected faces, predicts facial features and returns analyzed data. Args: path_image (str): Path to the input image. batch_size (int): Batch size for making predictions on the faces. Default is 8. fix_img_size (bool): If True, resizes the image to the size specified in reader. Default is False. return_img_data (bool): If True, returns all image data including tensors, otherwise only returns the faces. Default is False. include_tensors (bool): If True, removes tensors from the returned data object. Default is False. path_output (Optional[str]): Path where to save the image with detected faces. If None, the image is not saved. Default: None. Returns: Union[Response, ImageData]: If return_img_data is False, returns a Response object containing the faces and their facial features. If return_img_data is True, returns the entire ImageData object.",
"func":1
},
{
"ref":"facetorch.analyzer",
"url":1,
"doc":""
},
{
"ref":"facetorch.analyzer.FaceAnalyzer",
"url":1,
"doc":"FaceAnalyzer is the main class that reads images and runs face detection, tensor unification, as well as facial feature analysis prediction. It is the orchestrator responsible for initializing and running the following components: 1. Reader - reads the image and returns an ImageData object containing the image tensor. 2. Detector - wrapper around a neural network that detects faces. 3. Unifier - processor that unifies sizes of all faces and normalizes them between 0 and 1. 4. Predictor list - list of wrappers around models trained to analyze facial features for example, expressions. Args: cfg (OmegaConf): Config object with image reader, face detector, unifier and predictor configurations. Attributes: cfg (OmegaConf): Config object with image reader, face detector, unifier and predictor configurations. reader (BaseReader): Reader object that reads the image and returns an ImageData object containing the image tensor. detector (FaceDetector): FaceDetector object that wraps a neural network that detects faces. unifier (FaceUnifier): FaceUnifier object that unifies sizes of all faces and normalizes them between 0 and 1. predictors (Dict[str, FacePredictor]): Dict of FacePredictor objects that predict facial features. Key is the name of the predictor. logger (logging.Logger): Logger object that logs messages."
},
{
"ref":"facetorch.analyzer.FaceAnalyzer.run",
"url":1,
"doc":"Reads image, detects faces, unifies the detected faces, predicts facial features and returns analyzed data. Args: path_image (str): Path to the input image. batch_size (int): Batch size for making predictions on the faces. Default is 8. fix_img_size (bool): If True, resizes the image to the size specified in reader. Default is False. return_img_data (bool): If True, returns all image data including tensors, otherwise only returns the faces. Default is False. include_tensors (bool): If True, removes tensors from the returned data object. Default is False. path_output (Optional[str]): Path where to save the image with detected faces. If None, the image is not saved. Default: None. Returns: Union[Response, ImageData]: If return_img_data is False, returns a Response object containing the faces and their facial features. If return_img_data is True, returns the entire ImageData object.",
"func":1
},
{
"ref":"facetorch.analyzer.core",
"url":2,
"doc":""
},
{
"ref":"facetorch.analyzer.core.FaceAnalyzer",
"url":2,
"doc":"FaceAnalyzer is the main class that reads images and runs face detection, tensor unification, as well as facial feature analysis prediction. It is the orchestrator responsible for initializing and running the following components: 1. Reader - reads the image and returns an ImageData object containing the image tensor. 2. Detector - wrapper around a neural network that detects faces. 3. Unifier - processor that unifies sizes of all faces and normalizes them between 0 and 1. 4. Predictor list - list of wrappers around models trained to analyze facial features for example, expressions. Args: cfg (OmegaConf): Config object with image reader, face detector, unifier and predictor configurations. Attributes: cfg (OmegaConf): Config object with image reader, face detector, unifier and predictor configurations. reader (BaseReader): Reader object that reads the image and returns an ImageData object containing the image tensor. detector (FaceDetector): FaceDetector object that wraps a neural network that detects faces. unifier (FaceUnifier): FaceUnifier object that unifies sizes of all faces and normalizes them between 0 and 1. predictors (Dict[str, FacePredictor]): Dict of FacePredictor objects that predict facial features. Key is the name of the predictor. logger (logging.Logger): Logger object that logs messages."
},
{
"ref":"facetorch.analyzer.core.FaceAnalyzer.run",
"url":2,
"doc":"Reads image, detects faces, unifies the detected faces, predicts facial features and returns analyzed data. Args: path_image (str): Path to the input image. batch_size (int): Batch size for making predictions on the faces. Default is 8. fix_img_size (bool): If True, resizes the image to the size specified in reader. Default is False. return_img_data (bool): If True, returns all image data including tensors, otherwise only returns the faces. Default is False. include_tensors (bool): If True, removes tensors from the returned data object. Default is False. path_output (Optional[str]): Path where to save the image with detected faces. If None, the image is not saved. Default: None. Returns: Union[Response, ImageData]: If return_img_data is False, returns a Response object containing the faces and their facial features. If return_img_data is True, returns the entire ImageData object.",
"func":1
},
{
"ref":"facetorch.analyzer.detector",
"url":3,
"doc":""
},
{
"ref":"facetorch.analyzer.detector.FaceDetector",
"url":3,
"doc":"FaceDetector is a wrapper around a neural network model that is trained to detect faces. Args: downloader (BaseDownloader): Downloader that downloads the model. device (torch.device): Torch device cpu or cuda for the model. preprocessor (BaseDetPreProcessor): Preprocessor that runs before the model. postprocessor (BaseDetPostProcessor): Postprocessor that runs after the model."
},
{
"ref":"facetorch.analyzer.detector.FaceDetector.run",
"url":3,
"doc":"Detect all faces in the image. Args: ImageData: ImageData object containing the image tensor with values between 0 - 255 and shape (batch_size, channels, height, width). Returns: ImageData: Image data object with Detection tensors and detected Face objects.",
"func":1
},
{
"ref":"facetorch.analyzer.detector.FaceDetector.load_model",
"url":4,
"doc":"Loads the TorchScript model. Returns: Union[torch.jit.ScriptModule, torch.jit.TracedModule]: Loaded TorchScript model.",
"func":1
},
{
"ref":"facetorch.analyzer.detector.FaceDetector.inference",
"url":4,
"doc":"Inference the model with the given tensor. Args: tensor (torch.Tensor): Input tensor for the model. Returns: Union[torch.Tensor, Tuple[torch.Tensor : Output tensor or tuple of tensors.",
"func":1
},
{
"ref":"facetorch.analyzer.detector.core",
"url":5,
"doc":""
},
{
"ref":"facetorch.analyzer.detector.core.FaceDetector",
"url":5,
"doc":"FaceDetector is a wrapper around a neural network model that is trained to detect faces. Args: downloader (BaseDownloader): Downloader that downloads the model. device (torch.device): Torch device cpu or cuda for the model. preprocessor (BaseDetPreProcessor): Preprocessor that runs before the model. postprocessor (BaseDetPostProcessor): Postprocessor that runs after the model."
},
{
"ref":"facetorch.analyzer.detector.core.FaceDetector.run",
"url":5,
"doc":"Detect all faces in the image. Args: ImageData: ImageData object containing the image tensor with values between 0 - 255 and shape (batch_size, channels, height, width). Returns: ImageData: Image data object with Detection tensors and detected Face objects.",
"func":1
},
{
"ref":"facetorch.analyzer.detector.core.FaceDetector.load_model",
"url":4,
"doc":"Loads the TorchScript model. Returns: Union[torch.jit.ScriptModule, torch.jit.TracedModule]: Loaded TorchScript model.",
"func":1
},
{
"ref":"facetorch.analyzer.detector.core.FaceDetector.inference",
"url":4,
"doc":"Inference the model with the given tensor. Args: tensor (torch.Tensor): Input tensor for the model. Returns: Union[torch.Tensor, Tuple[torch.Tensor : Output tensor or tuple of tensors.",
"func":1
},
{
"ref":"facetorch.analyzer.detector.post",
"url":6,
"doc":""
},
{
"ref":"facetorch.analyzer.detector.post.BaseDetPostProcessor",
"url":6,
"doc":"Base class for detector post processors. All detector post processors should subclass it. All subclass should overwrite: - Methods: run , used for running the processing Args: device (torch.device): Torch device cpu or cuda. transform (transforms.Compose): Transform compose object to be applied to the image. optimize_transform (bool): Whether to optimize the transform."
},
{
"ref":"facetorch.analyzer.detector.post.BaseDetPostProcessor.run",
"url":6,
"doc":"Abstract method that runs the detector post processing functionality and returns the data object. Args: data (ImageData): ImageData object containing the image tensor. logits (Union[torch.Tensor, Tuple[torch.Tensor ): Output of the detector model. Returns: ImageData: Image data object with Detection tensors and detected Face objects.",
"func":1
},
{
"ref":"facetorch.analyzer.detector.post.BaseDetPostProcessor.optimize",
"url":4,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.detector.post.PriorBox",
"url":6,
"doc":"PriorBox class for generating prior boxes. Args: min_sizes (List[List[int ): List of list of minimum sizes for each feature map. steps (List[int]): List of steps for each feature map. clip (bool): Whether to clip the prior boxes to the image boundaries."
},
{
"ref":"facetorch.analyzer.detector.post.PriorBox.forward",
"url":6,
"doc":"Generate prior boxes for each feature map. Args: dims (Dimensions): Dimensions of the image. Returns: torch.Tensor: Tensor of prior boxes.",
"func":1
},
{
"ref":"facetorch.analyzer.detector.post.PostRetFace",
"url":6,
"doc":"Initialize the detector postprocessor. Modified from: https: github.com/biubug6/Pytorch_Retinaface. Args: transform (Compose): Composed Torch transform object. device (torch.device): Torch device cpu or cuda. optimize_transform (bool): Whether to optimize the transform. confidence_threshold (float): Confidence threshold for face detection. top_k (int): Top K faces to keep before NMS. nms_threshold (float): NMS threshold. keep_top_k (int): Keep top K faces after NMS. score_threshold (float): Score threshold for face detection. prior_box (PriorBox): PriorBox object. variance (List[float]): Prior box variance. reverse_colors (bool): Whether to reverse the colors of the image tensor from RGB to BGR or vice versa. If False, the colors remain unchanged. Default: False. expand_pixels (int): Number of pixels to expand the face location and tensor by. Default: 0."
},
{
"ref":"facetorch.analyzer.detector.post.PostRetFace.run",
"url":6,
"doc":"Run the detector postprocessor. Args: data (ImageData): ImageData object containing the image tensor. logits (Union[torch.Tensor, Tuple[torch.Tensor ): Output of the detector model. Returns: ImageData: Image data object with detection tensors and detected Face objects.",
"func":1
},
{
"ref":"facetorch.analyzer.detector.post.PostRetFace.optimize",
"url":4,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.detector.pre",
"url":7,
"doc":""
},
{
"ref":"facetorch.analyzer.detector.pre.BaseDetPreProcessor",
"url":7,
"doc":"Base class for detector pre processors. All detector pre processors should subclass it. All subclass should overwrite: - Methods: run , used for running the processing Args: device (torch.device): Torch device cpu or cuda. transform (transforms.Compose): Transform compose object to be applied to the image. optimize_transform (bool): Whether to optimize the transform."
},
{
"ref":"facetorch.analyzer.detector.pre.BaseDetPreProcessor.run",
"url":7,
"doc":"Abstract method that runs the detector pre processing functionality. Returns a batch of preprocessed face tensors. Args: data (ImageData): ImageData object containing the image tensor. Returns: ImageData: ImageData object containing the image tensor preprocessed for the detector.",
"func":1
},
{
"ref":"facetorch.analyzer.detector.pre.BaseDetPreProcessor.optimize",
"url":4,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.detector.pre.DetectorPreProcessor",
"url":7,
"doc":"Initialize the detector preprocessor. Args: transform (Compose): Composed Torch transform object. device (torch.device): Torch device cpu or cuda. optimize_transform (bool): Whether to optimize the transform. reverse_colors (bool): Whether to reverse the colors of the image tensor from RGB to BGR or vice versa. If False, the colors remain unchanged."
},
{
"ref":"facetorch.analyzer.detector.pre.DetectorPreProcessor.run",
"url":7,
"doc":"Run the detector preprocessor on the image tensor in BGR format and return the transformed image tensor. Args: data (ImageData): ImageData object containing the image tensor. Returns: ImageData: ImageData object containing the preprocessed image tensor.",
"func":1
},
{
"ref":"facetorch.analyzer.detector.pre.DetectorPreProcessor.optimize",
"url":4,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor",
"url":8,
"doc":""
},
{
"ref":"facetorch.analyzer.predictor.FacePredictor",
"url":8,
"doc":"FacePredictor is a wrapper around a neural network model that is trained to predict facial features. Args: downloader (BaseDownloader): Downloader that downloads the model. device (torch.device): Torch device cpu or cuda for the model. preprocessor (BasePredPostProcessor): Preprocessor that runs before the model. postprocessor (BasePredPostProcessor): Postprocessor that runs after the model."
},
{
"ref":"facetorch.analyzer.predictor.FacePredictor.run",
"url":8,
"doc":"Predicts facial features. Args: faces (torch.Tensor): Torch tensor containing a batch of faces with values between 0-1 and shape (batch_size, channels, height, width). Returns: (List[Prediction]): List of Prediction data objects. One for each face in the batch.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.FacePredictor.load_model",
"url":4,
"doc":"Loads the TorchScript model. Returns: Union[torch.jit.ScriptModule, torch.jit.TracedModule]: Loaded TorchScript model.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.FacePredictor.inference",
"url":4,
"doc":"Inference the model with the given tensor. Args: tensor (torch.Tensor): Input tensor for the model. Returns: Union[torch.Tensor, Tuple[torch.Tensor : Output tensor or tuple of tensors.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.core",
"url":9,
"doc":""
},
{
"ref":"facetorch.analyzer.predictor.core.FacePredictor",
"url":9,
"doc":"FacePredictor is a wrapper around a neural network model that is trained to predict facial features. Args: downloader (BaseDownloader): Downloader that downloads the model. device (torch.device): Torch device cpu or cuda for the model. preprocessor (BasePredPostProcessor): Preprocessor that runs before the model. postprocessor (BasePredPostProcessor): Postprocessor that runs after the model."
},
{
"ref":"facetorch.analyzer.predictor.core.FacePredictor.run",
"url":9,
"doc":"Predicts facial features. Args: faces (torch.Tensor): Torch tensor containing a batch of faces with values between 0-1 and shape (batch_size, channels, height, width). Returns: (List[Prediction]): List of Prediction data objects. One for each face in the batch.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.core.FacePredictor.load_model",
"url":4,
"doc":"Loads the TorchScript model. Returns: Union[torch.jit.ScriptModule, torch.jit.TracedModule]: Loaded TorchScript model.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.core.FacePredictor.inference",
"url":4,
"doc":"Inference the model with the given tensor. Args: tensor (torch.Tensor): Input tensor for the model. Returns: Union[torch.Tensor, Tuple[torch.Tensor : Output tensor or tuple of tensors.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.post",
"url":10,
"doc":""
},
{
"ref":"facetorch.analyzer.predictor.post.BasePredPostProcessor",
"url":10,
"doc":"Base class for predictor post processors. All predictor post processors should subclass it. All subclass should overwrite: - Methods: run , used for running the processing Args: device (torch.device): Torch device cpu or cuda. transform (transforms.Compose): Transform compose object to be applied to the image. optimize_transform (bool): Whether to optimize the transform. labels (List[str]): List of labels."
},
{
"ref":"facetorch.analyzer.predictor.post.BasePredPostProcessor.create_pred_list",
"url":10,
"doc":"Create a list of predictions. Args: preds (torch.Tensor): Tensor of predictions, shape (batch, _). indices (List[int]): List of label indices, one for each sample. Returns: List[Prediction]: List of predictions.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.post.BasePredPostProcessor.run",
"url":10,
"doc":"Abstract method that runs the predictor post processing functionality and returns a list of prediction data structures, one for each face in the batch. Args: preds (Union[torch.Tensor, Tuple[torch.Tensor ): Output of the predictor model. Returns: List[Prediction]: List of predictions.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.post.BasePredPostProcessor.optimize",
"url":4,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.post.PostArgMax",
"url":10,
"doc":"Initialize the predictor postprocessor that runs argmax on the prediction tensor and returns a list of prediction data structures. Args: transform (Compose): Composed Torch transform object. device (torch.device): Torch device cpu or cuda. optimize_transform (bool): Whether to optimize the transform using TorchScript. labels (List[str]): List of labels. dim (int): Dimension of the prediction."
},
{
"ref":"facetorch.analyzer.predictor.post.PostArgMax.run",
"url":10,
"doc":"Post-processes the prediction tensor using argmax and returns a list of prediction data structures, one for each face. Args: preds (torch.Tensor): Batch prediction tensor. Returns: List[Prediction]: List of prediction data structures containing the predicted labels and confidence scores for each face in the batch.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.post.PostArgMax.create_pred_list",
"url":10,
"doc":"Create a list of predictions. Args: preds (torch.Tensor): Tensor of predictions, shape (batch, _). indices (List[int]): List of label indices, one for each sample. Returns: List[Prediction]: List of predictions.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.post.PostArgMax.optimize",
"url":4,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.post.PostSigmoidBinary",
"url":10,
"doc":"Initialize the predictor postprocessor that runs sigmoid on the prediction tensor and returns a list of prediction data structures. Args: transform (Compose): Composed Torch transform object. device (torch.device): Torch device cpu or cuda. optimize_transform (bool): Whether to optimize the transform using TorchScript. labels (List[str]): List of labels. threshold (float): Probability threshold for positive class."
},
{
"ref":"facetorch.analyzer.predictor.post.PostSigmoidBinary.run",
"url":10,
"doc":"Post-processes the prediction tensor using argmax and returns a list of prediction data structures, one for each face. Args: preds (torch.Tensor): Batch prediction tensor. Returns: List[Prediction]: List of prediction data structures containing the predicted labelsand confidence scores for each face in the batch.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.post.PostSigmoidBinary.create_pred_list",
"url":10,
"doc":"Create a list of predictions. Args: preds (torch.Tensor): Tensor of predictions, shape (batch, _). indices (List[int]): List of label indices, one for each sample. Returns: List[Prediction]: List of predictions.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.post.PostSigmoidBinary.optimize",
"url":4,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.post.PostEmbedder",
"url":10,
"doc":"Initialize the predictor postprocessor that extracts the embedding from the prediction tensor and returns a list of prediction data structures. Args: transform (Compose): Composed Torch transform object. device (torch.device): Torch device cpu or cuda. optimize_transform (bool): Whether to optimize the transform using TorchScript. labels (List[str]): List of labels."
},
{
"ref":"facetorch.analyzer.predictor.post.PostEmbedder.run",
"url":10,
"doc":"Post-processes the prediction tensor using argmax and returns a list of prediction data structures, one for each face. Args: preds (torch.Tensor): Batch prediction tensor. Returns: List[Prediction]: List of prediction data structures containing the predicted labels and confidence scores for each face in the batch.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.post.PostEmbedder.create_pred_list",
"url":10,
"doc":"Create a list of predictions. Args: preds (torch.Tensor): Tensor of predictions, shape (batch, _). indices (List[int]): List of label indices, one for each sample. Returns: List[Prediction]: List of predictions.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.post.PostEmbedder.optimize",
"url":4,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.pre",
"url":11,
"doc":""
},
{
"ref":"facetorch.analyzer.predictor.pre.BasePredPreProcessor",
"url":11,
"doc":"Base class for predictor pre processors. All predictor pre processors should subclass it. All subclass should overwrite: - Methods: run , used for running the processing Args: device (torch.device): Torch device cpu or cuda. transform (transforms.Compose): Transform compose object to be applied to the image. optimize_transform (bool): Whether to optimize the transform."
},
{
"ref":"facetorch.analyzer.predictor.pre.BasePredPreProcessor.run",
"url":11,
"doc":"Abstract method that runs the predictor pre processing functionality and returns a batch of preprocessed face tensors. Args: faces (torch.Tensor): Batch of face tensors with shape (batch, channels, height, width). Returns: torch.Tensor: Batch of preprocessed face tensors with shape (batch, channels, height, width).",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.pre.BasePredPreProcessor.optimize",
"url":4,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.pre.PredictorPreProcessor",
"url":11,
"doc":"Torch transform based pre-processor that is applied to face tensors before they are passed to the predictor model. Args: transform (transforms.Compose): Composed Torch transform object. device (torch.device): Torch device cpu or cuda. optimize_transform (bool): Whether to optimize the transform. reverse_colors (bool): Whether to reverse the colors of the image tensor"
},
{
"ref":"facetorch.analyzer.predictor.pre.PredictorPreProcessor.run",
"url":11,
"doc":"Runs the trasform on a batch of face tensors. Args: faces (torch.Tensor): Batch of face tensors. Returns: torch.Tensor: Batch of preprocessed face tensors.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.pre.PredictorPreProcessor.optimize",
"url":4,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.reader",
"url":12,
"doc":""
},
{
"ref":"facetorch.analyzer.reader.ImageReader",
"url":12,
"doc":"ImageReader is a wrapper around a functionality for reading images by Torchvision. Args: transform (torchvision.transforms.Compose): Transform compose object to be applied to the image, if fix_image_size is True. device (torch.device): Torch device cpu or cuda object. optimize_transform (bool): Whether to optimize the transforms that are: resizing the image to a fixed size."
},
{
"ref":"facetorch.analyzer.reader.ImageReader.run",
"url":12,
"doc":"Reads an image from a path and returns a tensor of the image with values between 0-255 and shape (batch, channels, height, width). The order of color channels is RGB. PyTorch and Torchvision are used to read the image. Args: path_image (str): Path to the image. fix_img_size (bool): Whether to resize the image to a fixed size. If False, the size_portrait and size_landscape are ignored. Default is False. Returns: ImageData: ImageData object with image tensor and pil Image.",
"func":1
},
{
"ref":"facetorch.analyzer.reader.ImageReader.optimize",
"url":4,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.reader.core",
"url":13,
"doc":""
},
{
"ref":"facetorch.analyzer.reader.core.ImageReader",
"url":13,
"doc":"ImageReader is a wrapper around a functionality for reading images by Torchvision. Args: transform (torchvision.transforms.Compose): Transform compose object to be applied to the image, if fix_image_size is True. device (torch.device): Torch device cpu or cuda object. optimize_transform (bool): Whether to optimize the transforms that are: resizing the image to a fixed size."
},
{
"ref":"facetorch.analyzer.reader.core.ImageReader.run",
"url":13,
"doc":"Reads an image from a path and returns a tensor of the image with values between 0-255 and shape (batch, channels, height, width). The order of color channels is RGB. PyTorch and Torchvision are used to read the image. Args: path_image (str): Path to the image. fix_img_size (bool): Whether to resize the image to a fixed size. If False, the size_portrait and size_landscape are ignored. Default is False. Returns: ImageData: ImageData object with image tensor and pil Image.",
"func":1
},
{
"ref":"facetorch.analyzer.reader.core.ImageReader.optimize",
"url":4,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.unifier",
"url":14,
"doc":""
},
{
"ref":"facetorch.analyzer.unifier.FaceUnifier",
"url":14,
"doc":"FaceUnifier is a transform based processor that can unify sizes of all faces and normalize them between 0 and 1. Args: transform (Compose): Composed Torch transform object. device (torch.device): Torch device cpu or cuda object. optimize_transform (bool): Whether to optimize the transform."
},
{
"ref":"facetorch.analyzer.unifier.FaceUnifier.run",
"url":14,
"doc":"Runs unifying transform on each face tensor one by one. Args: data (ImageData): ImageData object containing the face tensors. Returns: ImageData: ImageData object containing the unified face tensors normalized between 0 and 1.",
"func":1
},
{
"ref":"facetorch.analyzer.unifier.FaceUnifier.optimize",
"url":4,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.unifier.core",
"url":15,
"doc":""
},
{
"ref":"facetorch.analyzer.unifier.core.FaceUnifier",
"url":15,
"doc":"FaceUnifier is a transform based processor that can unify sizes of all faces and normalize them between 0 and 1. Args: transform (Compose): Composed Torch transform object. device (torch.device): Torch device cpu or cuda object. optimize_transform (bool): Whether to optimize the transform."
},
{
"ref":"facetorch.analyzer.unifier.core.FaceUnifier.run",
"url":15,
"doc":"Runs unifying transform on each face tensor one by one. Args: data (ImageData): ImageData object containing the face tensors. Returns: ImageData: ImageData object containing the unified face tensors normalized between 0 and 1.",
"func":1
},
{
"ref":"facetorch.analyzer.unifier.core.FaceUnifier.optimize",
"url":4,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.base",
"url":4,
"doc":""
},
{
"ref":"facetorch.base.BaseProcessor",
"url":4,
"doc":"Base class for processors. All data pre and post processors should subclass it. All subclass should overwrite: - Methods: run , used for running the processing functionality. Args: device (torch.device): Torch device cpu or cuda. transform (transforms.Compose): Transform compose object to be applied to the image. optimize_transform (bool): Whether to optimize the transform."
},
{
"ref":"facetorch.base.BaseProcessor.optimize",
"url":4,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.base.BaseProcessor.run",
"url":4,
"doc":"Abstract method that should implement a tensor processing functionality",
"func":1
},
{
"ref":"facetorch.base.BaseReader",
"url":4,
"doc":"Base class for image reader. All image readers should subclass it. All subclass should overwrite: - Methods: run , used for running the reading process and return a tensor. Args: transform (transforms.Compose): Transform to be applied to the image. device (torch.device): Torch device cpu or cuda. optimize_transform (bool): Whether to optimize the transforms that are resizing the image to a fixed size."
},
{
"ref":"facetorch.base.BaseReader.run",
"url":4,
"doc":"Abstract method that reads an image from a path and returns a data object containing a tensor of the image with shape (batch, channels, height, width). Args: path (str): Path to the image. Returns: ImageData: ImageData object with the image tensor.",
"func":1
},
{
"ref":"facetorch.base.BaseReader.optimize",
"url":4,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.base.BaseDownloader",
"url":4,
"doc":"Base class for downloaders. All downloaders should subclass it. All subclass should overwrite: - Methods: run , supporting to run the download functionality. Args: file_id (str): ID of the hosted file (e.g. Google Drive File ID). path_local (str): The file is downloaded to this local path."
},
{
"ref":"facetorch.base.BaseDownloader.run",
"url":4,
"doc":"Abstract method that should implement the download functionality",
"func":1
},
{
"ref":"facetorch.base.BaseModel",
"url":4,
"doc":"Base class for torch models. All detectors and predictors should subclass it. All subclass should overwrite: - Methods: run , supporting to make detections and predictions with the model. Args: downloader (BaseDownloader): Downloader for the model. device (torch.device): Torch device cpu or cuda. Attributes: model (torch.jit.ScriptModule or torch.jit.TracedModule): Loaded TorchScript model."
},
{
"ref":"facetorch.base.BaseModel.load_model",
"url":4,
"doc":"Loads the TorchScript model. Returns: Union[torch.jit.ScriptModule, torch.jit.TracedModule]: Loaded TorchScript model.",
"func":1
},
{
"ref":"facetorch.base.BaseModel.inference",
"url":4,
"doc":"Inference the model with the given tensor. Args: tensor (torch.Tensor): Input tensor for the model. Returns: Union[torch.Tensor, Tuple[torch.Tensor : Output tensor or tuple of tensors.",
"func":1
},
{
"ref":"facetorch.base.BaseModel.run",
"url":4,
"doc":"Abstract method for making the predictions. Example pipeline: - self.preprocessor.run - self.inference - self.postprocessor.run",
"func":1
},
{
"ref":"facetorch.datastruct",
"url":16,
"doc":""
},
{
"ref":"facetorch.datastruct.Dimensions",
"url":16,
"doc":"Data class for image dimensions. Attributes: height (int): Image height. width (int): Image width."
},
{
"ref":"facetorch.datastruct.Dimensions.height",
"url":16,
"doc":""
},
{
"ref":"facetorch.datastruct.Dimensions.width",
"url":16,
"doc":""
},
{
"ref":"facetorch.datastruct.Location",
"url":16,
"doc":"Data class for face location. Attributes: x1 (int): x1 coordinate x2 (int): x2 coordinate y1 (int): y1 coordinate y2 (int): y2 coordinate"
},
{
"ref":"facetorch.datastruct.Location.x1",
"url":16,
"doc":""
},
{
"ref":"facetorch.datastruct.Location.x2",
"url":16,
"doc":""
},
{
"ref":"facetorch.datastruct.Location.y1",
"url":16,
"doc":""
},
{
"ref":"facetorch.datastruct.Location.y2",
"url":16,
"doc":""
},
{
"ref":"facetorch.datastruct.Location.form_square",
"url":16,
"doc":"Form a square from the location. Returns: None",
"func":1
},
{
"ref":"facetorch.datastruct.Location.expand",
"url":16,
"doc":"Expand the location while keeping the center. Args: amount (int): Amount of pixels to expand the location by. Returns: None",
"func":1
},
{
"ref":"facetorch.datastruct.Prediction",
"url":16,
"doc":"Data class for face prediction results. Attributes: label (str): Label of the face given by predictor. logits (torch.Tensor): Output of the predictor model for the face."
},
{
"ref":"facetorch.datastruct.Prediction.label",
"url":16,
"doc":""
},
{
"ref":"facetorch.datastruct.Prediction.logits",
"url":16,
"doc":""
},
{
"ref":"facetorch.datastruct.Detection",
"url":16,
"doc":"Data class for detector output. Attributes: loc (torch.Tensor): Locations of faces conf (torch.Tensor): Confidences of faces landmarks (torch.Tensor): Landmarks of faces boxes (torch.Tensor): Bounding boxes of faces dets (torch.Tensor): Detections of faces"
},
{
"ref":"facetorch.datastruct.Detection.loc",
"url":16,
"doc":""
},
{
"ref":"facetorch.datastruct.Detection.conf",
"url":16,
"doc":""
},
{
"ref":"facetorch.datastruct.Detection.landmarks",
"url":16,
"doc":""
},
{
"ref":"facetorch.datastruct.Detection.boxes",
"url":16,
"doc":""
},
{
"ref":"facetorch.datastruct.Detection.dets",
"url":16,
"doc":""
},
{
"ref":"facetorch.datastruct.Face",
"url":16,
"doc":"Data class for face attributes. Attributes: indx (int): Index of the face. loc (Location): Location of the face in the image. dims (Dimensions): Dimensions of the face (height, width). tensor (torch.Tensor): Face tensor. ratio (float): Ratio of the face area to the image area. preds (Dict[str, Prediction]): Predictions of the face given by predictor set."
},
{
"ref":"facetorch.datastruct.Face.indx",
"url":16,
"doc":""
},
{
"ref":"facetorch.datastruct.Face.loc",
"url":16,
"doc":""
},
{
"ref":"facetorch.datastruct.Face.dims",
"url":16,
"doc":""
},
{
"ref":"facetorch.datastruct.Face.tensor",
"url":16,
"doc":""
},
{
"ref":"facetorch.datastruct.Face.ratio",
"url":16,
"doc":""
},
{
"ref":"facetorch.datastruct.Face.preds",
"url":16,
"doc":""
},
{
"ref":"facetorch.datastruct.ImageData",
"url":16,
"doc":"The main data class used for passing data between the different facetorch modules. Attributes: path (str): Path to the image. img (torch.Tensor): Original image tensor. tensor (torch.Tensor): Processed image tensor. dims (Dimensions): Dimensions of the image (height, width). det (Detection): Detection data. faces (Dict[int, Face]): Dictionary of faces. version (int): Version of the image."
},
{
"ref":"facetorch.datastruct.ImageData.path",
"url":16,
"doc":""
},
{
"ref":"facetorch.datastruct.ImageData.img",
"url":16,
"doc":""
},
{
"ref":"facetorch.datastruct.ImageData.tensor",
"url":16,
"doc":""
},
{
"ref":"facetorch.datastruct.ImageData.dims",
"url":16,
"doc":""
},
{
"ref":"facetorch.datastruct.ImageData.det",
"url":16,
"doc":""
},
{
"ref":"facetorch.datastruct.ImageData.faces",
"url":16,
"doc":""
},
{
"ref":"facetorch.datastruct.ImageData.version",
"url":16,
"doc":""
},
{
"ref":"facetorch.datastruct.ImageData.add_preds",
"url":16,
"doc":"Adds a list of predictions to the data object. Args: preds_list (List[Prediction]): List of predictions. predictor_name (str): Name of the predictor. face_offset (int): Offset of the face index. Returns: None",
"func":1
},
{
"ref":"facetorch.datastruct.ImageData.reset_img",
"url":16,
"doc":"Reset the original image tensor to empty state.",
"func":1
},
{
"ref":"facetorch.datastruct.ImageData.reset_tensor",
"url":16,
"doc":"Reset the processed image tensor to empty state.",
"func":1
},
{
"ref":"facetorch.datastruct.ImageData.reset_face_tensors",
"url":16,
"doc":"Reset the face tensors to empty state.",
"func":1
},
{
"ref":"facetorch.datastruct.ImageData.reset_face_pred_tensors",
"url":16,
"doc":"Reset the face prediction tensors to empty state.",
"func":1
},
{
"ref":"facetorch.datastruct.ImageData.reset_det_tensors",
"url":16,
"doc":"Reset the detection object to empty state.",
"func":1
},
{
"ref":"facetorch.datastruct.ImageData.reset_tensors",
"url":16,
"doc":"Reset the tensors to empty state.",
"func":1
},
{
"ref":"facetorch.datastruct.ImageData.set_dims",
"url":16,
"doc":"Set the dimensions attribute from the tensor attribute.",
"func":1
},
{
"ref":"facetorch.datastruct.ImageData.aggregate_loc_tensor",
"url":16,
"doc":"Aggregates the location tensor from all faces. Returns: torch.Tensor: Aggregated location tensor for drawing purposes.",
"func":1
},
{
"ref":"facetorch.datastruct.Response",
"url":16,
"doc":"Data class for response data that is a subset of ImageData. Attributes: faces (Dict[int, Face]): Dictionary of faces. version (int): Version of the image."
},
{
"ref":"facetorch.datastruct.Response.faces",
"url":16,
"doc":""
},
{
"ref":"facetorch.datastruct.Response.version",
"url":16,
"doc":""
},
{
"ref":"facetorch.downloader",
"url":17,
"doc":""
},
{
"ref":"facetorch.downloader.DownloaderGDrive",
"url":17,
"doc":"Downloader for Google Drive files. Args: file_id (str): ID of the file hosted on Google Drive. path_local (str): The file is downloaded to this local path."
},
{
"ref":"facetorch.downloader.DownloaderGDrive.run",
"url":17,
"doc":"Downloads a file from Google Drive.",
"func":1
},
{
"ref":"facetorch.logger",
"url":18,
"doc":""
},
{
"ref":"facetorch.logger.LoggerJsonFile",
"url":18,
"doc":"Logger in json format that writes to a file and console. Args: name (str): Name of the logger. level (str): Level of the logger. path_file (str): Path to the log file. json_format (str): Format of the log record. Attributes: logger (logging.Logger): Logger object."
},
{
"ref":"facetorch.logger.LoggerJsonFile.configure",
"url":18,
"doc":"Configures the logger.",
"func":1
},
{
"ref":"facetorch.transforms",
"url":19,
"doc":""
},
{
"ref":"facetorch.transforms.script_transform",
"url":19,
"doc":"Convert the composed transform to a TorchScript module. Args: transform (transforms.Compose): Transform compose object to be scripted. Returns: Union[torch.jit.ScriptModule, torch.jit.ScriptFunction]: Scripted transform.",
"func":1
},
{
"ref":"facetorch.transforms.SquarePad",
"url":19,
"doc":"SquarePad is a transform that pads the image to a square shape. It is initialized as a torch.nn.Module."
},
{
"ref":"facetorch.transforms.SquarePad.forward",
"url":19,
"doc":"Pads a tensor to a square. Args: tensor (torch.Tensor): tensor to pad. Returns: torch.Tensor: Padded tensor.",
"func":1
},
{
"ref":"facetorch.utils",
"url":20,
"doc":""
},
{
"ref":"facetorch.utils.rgb2bgr",
"url":20,
"doc":"Converts a batch of RGB tensors to BGR tensors or vice versa. Args: tensor (torch.Tensor): Batch of RGB (or BGR) channeled tensors with shape (dim0, channels, dim2, dim3) Returns: torch.Tensor: Batch of BGR (or RGB) tensors with shape (dim0, channels, dim2, dim3).",
"func":1
},
{
"ref":"facetorch.utils.draw_boxes_and_save",
"url":20,
"doc":"Draws boxes on an image and saves it to a file. Args: data (ImageData): ImageData object containing the image tensor, detections, and faces. path_output (str): Path to the output file. Returns: None",
"func":1
},
{
"ref":"facetorch.utils.fix_transform_list_attr",
"url":20,
"doc":"Fix the transform attributes by converting the listconfig to a list. This enables to optimize the transform using TorchScript. Args: transform (torchvision.transforms.Compose): Transform to be fixed. Returns: torchvision.transforms.Compose: Fixed transform.",
"func":1
}
]