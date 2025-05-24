URLS=[
"facetorch/index.html",
"facetorch/base.html",
"facetorch/datastruct.html",
"facetorch/transforms.html",
"facetorch/utils.html",
"facetorch/downloader.html",
"facetorch/logger.html",
"facetorch/analyzer/index.html",
"facetorch/analyzer/detector/index.html",
"facetorch/analyzer/detector/post.html",
"facetorch/analyzer/detector/pre.html",
"facetorch/analyzer/detector/core.html",
"facetorch/analyzer/unifier/index.html",
"facetorch/analyzer/unifier/core.html",
"facetorch/analyzer/predictor/index.html",
"facetorch/analyzer/predictor/post.html",
"facetorch/analyzer/predictor/pre.html",
"facetorch/analyzer/predictor/core.html",
"facetorch/analyzer/utilizer/index.html",
"facetorch/analyzer/utilizer/save.html",
"facetorch/analyzer/utilizer/draw.html",
"facetorch/analyzer/utilizer/align.html",
"facetorch/analyzer/reader/index.html",
"facetorch/analyzer/reader/core.html",
"facetorch/analyzer/core.html"
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
"doc":"FaceAnalyzer is the main class that reads images, runs face detection, tensor unification and facial feature prediction. It also draws bounding boxes and facial landmarks over the image. The following components are used: 1. Reader - reads the image and returns an ImageData object containing the image tensor. 2. Detector - wrapper around a neural network that detects faces. 3. Unifier - processor that unifies sizes of all faces and normalizes them between 0 and 1. 4. Predictor dict - dict of wrappers around neural networks trained to analyze facial features. 5. Utilizer dict - dict of utilizer processors that can for example extract 3D face landmarks or draw boxes over the image. Args: cfg (OmegaConf): Config object with image reader, face detector, unifier and predictor configurations. Attributes: cfg (OmegaConf): Config object with image reader, face detector, unifier and predictor configurations. reader (BaseReader): Reader object that reads the image and returns an ImageData object containing the image tensor. detector (FaceDetector): FaceDetector object that wraps a neural network that detects faces. unifier (FaceUnifier): FaceUnifier object that unifies sizes of all faces and normalizes them between 0 and 1. predictors (Dict[str, FacePredictor]): Dict of FacePredictor objects that predict facial features. Key is the name of the predictor. utilizers (Dict[str, FaceUtilizer]): Dict of FaceUtilizer objects that can extract 3D face landmarks, draw boxes over the image, etc. Key is the name of the utilizer. logger (logging.Logger): Logger object that logs messages to the console or to a file."
},
{
"ref":"facetorch.FaceAnalyzer.run",
"url":0,
"doc":"Reads image, detects faces, unifies the detected faces, predicts facial features and returns analyzed data. Args: image_source (Optional[Union[str, torch.Tensor, np.ndarray, bytes, Image.Image ): Input to be analyzed. If None, path_image or tensor must be provided. Default: None. path_image (Optional[str]): Path to the image to be analyzed. If None, tensor must be provided. Default: None. batch_size (int): Batch size for making predictions on the faces. Default is 8. fix_img_size (bool): If True, resizes the image to the size specified in reader. Default is False. return_img_data (bool): If True, returns all image data including tensors, otherwise only returns the faces. Default is False. include_tensors (bool): If True, removes tensors from the returned data object. Default is False. path_output (Optional[str]): Path where to save the image with detected faces. If None, the image is not saved. Default: None. tensor (Optional[torch.Tensor]): Image tensor to be analyzed. If None, path_image must be provided. Default: None. Returns: Union[Response, ImageData]: If return_img_data is False, returns a Response object containing the faces and their facial features. If return_img_data is True, returns the entire ImageData object.",
"func":1
},
{
"ref":"facetorch.base",
"url":1,
"doc":""
},
{
"ref":"facetorch.base.BaseProcessor",
"url":1,
"doc":"Base class for processors. All data pre and post processors should subclass it. All subclass should overwrite: - Methods: run , used for running the processing functionality. Args: device (torch.device): Torch device cpu or cuda. transform (transforms.Compose): Transform compose object to be applied to the image. optimize_transform (bool): Whether to optimize the transform."
},
{
"ref":"facetorch.base.BaseProcessor.optimize",
"url":1,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.base.BaseProcessor.run",
"url":1,
"doc":"Abstract method that should implement a tensor processing functionality",
"func":1
},
{
"ref":"facetorch.base.BaseReader",
"url":1,
"doc":"Base class for image reader. All image readers should subclass it. All subclass should overwrite: - Methods: run , used for running the reading process and return a tensor. Args: transform (transforms.Compose): Transform to be applied to the image. device (torch.device): Torch device cpu or cuda. optimize_transform (bool): Whether to optimize the transforms that are resizing the image to a fixed size."
},
{
"ref":"facetorch.base.BaseReader.run",
"url":1,
"doc":"Abstract method that reads an image from a path and returns a data object containing a tensor of the image with shape (batch, channels, height, width). Args: path (str): Path to the image. Returns: ImageData: ImageData object with the image tensor.",
"func":1
},
{
"ref":"facetorch.base.BaseReader.process_tensor",
"url":1,
"doc":"Read a tensor and return a data object containing a tensor of the image with shape (batch, channels, height, width). Args: tensor (torch.Tensor): Tensor of a single image with RGB values between 0-255 and shape (channels, height, width). fix_img_size (bool): Whether to resize the image to a fixed size. If False, the size_portrait and size_landscape are ignored. Default is False.",
"func":1
},
{
"ref":"facetorch.base.BaseReader.optimize",
"url":1,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.base.BaseDownloader",
"url":1,
"doc":"Base class for downloaders. All downloaders should subclass it. All subclass should overwrite: - Methods: run , supporting to run the download functionality. Args: file_id (str): ID of the hosted file (e.g. Google Drive File ID). path_local (str): The file is downloaded to this local path."
},
{
"ref":"facetorch.base.BaseDownloader.run",
"url":1,
"doc":"Abstract method that should implement the download functionality",
"func":1
},
{
"ref":"facetorch.base.BaseModel",
"url":1,
"doc":"Base class for torch models. All detectors and predictors should subclass it. All subclass should overwrite: - Methods: run , supporting to make detections and predictions with the model. Args: downloader (BaseDownloader): Downloader for the model. device (torch.device): Torch device cpu or cuda. Attributes: model (torch.jit.ScriptModule or torch.jit.TracedModule): Loaded TorchScript model."
},
{
"ref":"facetorch.base.BaseModel.load_model",
"url":1,
"doc":"Loads the TorchScript model. Returns: Union[torch.jit.ScriptModule, torch.jit.TracedModule]: Loaded TorchScript model.",
"func":1
},
{
"ref":"facetorch.base.BaseModel.inference",
"url":1,
"doc":"Inference the model with the given tensor. Args: tensor (torch.Tensor): Input tensor for the model. Returns: Union[torch.Tensor, Tuple[torch.Tensor : Output tensor or tuple of tensors.",
"func":1
},
{
"ref":"facetorch.base.BaseModel.run",
"url":1,
"doc":"Abstract method for making the predictions. Example pipeline: - self.preprocessor.run - self.inference - self.postprocessor.run",
"func":1
},
{
"ref":"facetorch.base.BaseUtilizer",
"url":1,
"doc":"BaseUtilizer is a processor that takes ImageData as input to do any kind of work that requires model predictions for example, drawing, summarizing, etc. Args: transform (Compose): Composed Torch transform object. device (torch.device): Torch device cpu or cuda object. optimize_transform (bool): Whether to optimize the transform."
},
{
"ref":"facetorch.base.BaseUtilizer.run",
"url":1,
"doc":"Runs utility function on the ImageData object. Args: data (ImageData): ImageData object containing most of the data including the predictions. Returns: ImageData: ImageData object containing the same data as input or modified object.",
"func":1
},
{
"ref":"facetorch.base.BaseUtilizer.optimize",
"url":1,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.datastruct",
"url":2,
"doc":""
},
{
"ref":"facetorch.datastruct.Dimensions",
"url":2,
"doc":"Data class for image dimensions. Attributes: height (int): Image height. width (int): Image width."
},
{
"ref":"facetorch.datastruct.Dimensions.height",
"url":2,
"doc":""
},
{
"ref":"facetorch.datastruct.Dimensions.width",
"url":2,
"doc":""
},
{
"ref":"facetorch.datastruct.Location",
"url":2,
"doc":"Data class for face location. Attributes: x1 (int): x1 coordinate x2 (int): x2 coordinate y1 (int): y1 coordinate y2 (int): y2 coordinate"
},
{
"ref":"facetorch.datastruct.Location.x1",
"url":2,
"doc":""
},
{
"ref":"facetorch.datastruct.Location.x2",
"url":2,
"doc":""
},
{
"ref":"facetorch.datastruct.Location.y1",
"url":2,
"doc":""
},
{
"ref":"facetorch.datastruct.Location.y2",
"url":2,
"doc":""
},
{
"ref":"facetorch.datastruct.Location.form_square",
"url":2,
"doc":"Form a square from the location. Returns: None",
"func":1
},
{
"ref":"facetorch.datastruct.Location.expand",
"url":2,
"doc":"Expand the location while keeping the center. Args: amount (float): Amount to expand the location by in multiples of the original size. Returns: None",
"func":1
},
{
"ref":"facetorch.datastruct.Prediction",
"url":2,
"doc":"Data class for face prediction results and derivatives. Attributes: label (str): Label of the face given by predictor. logits (torch.Tensor): Output of the predictor model for the face. other (Dict): Any other predictions and derivatives for the face."
},
{
"ref":"facetorch.datastruct.Prediction.label",
"url":2,
"doc":""
},
{
"ref":"facetorch.datastruct.Prediction.logits",
"url":2,
"doc":""
},
{
"ref":"facetorch.datastruct.Prediction.other",
"url":2,
"doc":""
},
{
"ref":"facetorch.datastruct.Detection",
"url":2,
"doc":"Data class for detector output. Attributes: loc (torch.Tensor): Locations of faces conf (torch.Tensor): Confidences of faces landmarks (torch.Tensor): Landmarks of faces boxes (torch.Tensor): Bounding boxes of faces dets (torch.Tensor): Detections of faces"
},
{
"ref":"facetorch.datastruct.Detection.loc",
"url":2,
"doc":""
},
{
"ref":"facetorch.datastruct.Detection.conf",
"url":2,
"doc":""
},
{
"ref":"facetorch.datastruct.Detection.landmarks",
"url":2,
"doc":""
},
{
"ref":"facetorch.datastruct.Detection.boxes",
"url":2,
"doc":""
},
{
"ref":"facetorch.datastruct.Detection.dets",
"url":2,
"doc":""
},
{
"ref":"facetorch.datastruct.Face",
"url":2,
"doc":"Data class for face attributes. Attributes: indx (int): Index of the face. loc (Location): Location of the face in the image. dims (Dimensions): Dimensions of the face (height, width). tensor (torch.Tensor): Face tensor. ratio (float): Ratio of the face area to the image area. preds (Dict[str, Prediction]): Predictions of the face given by predictor set."
},
{
"ref":"facetorch.datastruct.Face.indx",
"url":2,
"doc":""
},
{
"ref":"facetorch.datastruct.Face.loc",
"url":2,
"doc":""
},
{
"ref":"facetorch.datastruct.Face.dims",
"url":2,
"doc":""
},
{
"ref":"facetorch.datastruct.Face.tensor",
"url":2,
"doc":""
},
{
"ref":"facetorch.datastruct.Face.ratio",
"url":2,
"doc":""
},
{
"ref":"facetorch.datastruct.Face.preds",
"url":2,
"doc":""
},
{
"ref":"facetorch.datastruct.ImageData",
"url":2,
"doc":"The main data class used for passing data between the different facetorch modules. Attributes: path_input (str): Path to the input image. path_output (str): Path to the output image where the resulting image is saved. img (torch.Tensor): Original image tensor used for drawing purposes. tensor (torch.Tensor): Processed image tensor. dims (Dimensions): Dimensions of the image (height, width). det (Detection): Detection data given by the detector. faces (List[Face]): List of faces in the image. version (str): Version of the facetorch library."
},
{
"ref":"facetorch.datastruct.ImageData.path_input",
"url":2,
"doc":""
},
{
"ref":"facetorch.datastruct.ImageData.path_output",
"url":2,
"doc":""
},
{
"ref":"facetorch.datastruct.ImageData.img",
"url":2,
"doc":""
},
{
"ref":"facetorch.datastruct.ImageData.tensor",
"url":2,
"doc":""
},
{
"ref":"facetorch.datastruct.ImageData.dims",
"url":2,
"doc":""
},
{
"ref":"facetorch.datastruct.ImageData.det",
"url":2,
"doc":""
},
{
"ref":"facetorch.datastruct.ImageData.faces",
"url":2,
"doc":""
},
{
"ref":"facetorch.datastruct.ImageData.version",
"url":2,
"doc":""
},
{
"ref":"facetorch.datastruct.ImageData.add_preds",
"url":2,
"doc":"Adds a list of predictions to the data object. Args: preds_list (List[Prediction]): List of predictions. predictor_name (str): Name of the predictor. face_offset (int): Offset of the face index where the predictions are added. Returns: None",
"func":1
},
{
"ref":"facetorch.datastruct.ImageData.reset_img",
"url":2,
"doc":"Reset the original image tensor to empty state.",
"func":1
},
{
"ref":"facetorch.datastruct.ImageData.reset_tensor",
"url":2,
"doc":"Reset the processed image tensor to empty state.",
"func":1
},
{
"ref":"facetorch.datastruct.ImageData.reset_face_tensors",
"url":2,
"doc":"Reset the face tensors to empty state.",
"func":1
},
{
"ref":"facetorch.datastruct.ImageData.reset_face_pred_tensors",
"url":2,
"doc":"Reset the face prediction tensors to empty state.",
"func":1
},
{
"ref":"facetorch.datastruct.ImageData.reset_det_tensors",
"url":2,
"doc":"Reset the detection object to empty state.",
"func":1
},
{
"ref":"facetorch.datastruct.ImageData.reset_tensors",
"url":2,
"doc":"Reset the tensors to empty state.",
"func":1
},
{
"ref":"facetorch.datastruct.ImageData.set_dims",
"url":2,
"doc":"Set the dimensions attribute from the tensor attribute.",
"func":1
},
{
"ref":"facetorch.datastruct.ImageData.aggregate_loc_tensor",
"url":2,
"doc":"Aggregates the location tensor from all faces. Returns: torch.Tensor: Aggregated location tensor for drawing purposes.",
"func":1
},
{
"ref":"facetorch.datastruct.Response",
"url":2,
"doc":"Data class for response data, which is a subset of ImageData. Attributes: faces (List[Face]): List of faces in the image. version (str): Version of the facetorch library."
},
{
"ref":"facetorch.datastruct.Response.faces",
"url":2,
"doc":""
},
{
"ref":"facetorch.datastruct.Response.version",
"url":2,
"doc":""
},
{
"ref":"facetorch.transforms",
"url":3,
"doc":""
},
{
"ref":"facetorch.transforms.script_transform",
"url":3,
"doc":"Convert the composed transform to a TorchScript module. Args: transform (transforms.Compose): Transform compose object to be scripted. Returns: Union[torch.jit.ScriptModule, torch.jit.ScriptFunction]: Scripted transform.",
"func":1
},
{
"ref":"facetorch.transforms.SquarePad",
"url":3,
"doc":"SquarePad is a transform that pads the image to a square shape. It is initialized as a torch.nn.Module."
},
{
"ref":"facetorch.transforms.SquarePad.forward",
"url":3,
"doc":"Pads a tensor to a square. Args: tensor (torch.Tensor): tensor to pad. Returns: torch.Tensor: Padded tensor.",
"func":1
},
{
"ref":"facetorch.utils",
"url":4,
"doc":""
},
{
"ref":"facetorch.utils.rgb2bgr",
"url":4,
"doc":"Converts a batch of RGB tensors to BGR tensors or vice versa. Args: tensor (torch.Tensor): Batch of RGB (or BGR) channeled tensors with shape (dim0, channels, dim2, dim3) Returns: torch.Tensor: Batch of BGR (or RGB) tensors with shape (dim0, channels, dim2, dim3).",
"func":1
},
{
"ref":"facetorch.utils.fix_transform_list_attr",
"url":4,
"doc":"Fix the transform attributes by converting the listconfig to a list. This enables to optimize the transform using TorchScript. Args: transform (torchvision.transforms.Compose): Transform to be fixed. Returns: torchvision.transforms.Compose: Fixed transform.",
"func":1
},
{
"ref":"facetorch.downloader",
"url":5,
"doc":""
},
{
"ref":"facetorch.downloader.DownloaderGDrive",
"url":5,
"doc":"Downloader for Google Drive files. Args: file_id (str): ID of the file hosted on Google Drive. path_local (str): The file is downloaded to this local path."
},
{
"ref":"facetorch.downloader.DownloaderGDrive.run",
"url":5,
"doc":"Downloads a file from Google Drive.",
"func":1
},
{
"ref":"facetorch.downloader.DownloaderHuggingFace",
"url":5,
"doc":"Downloader for HuggingFace Hub files. This downloader retrieves model files from the HuggingFace Hub, serving as an alternative to Google Drive for storing and accessing facetorch models. This allows for better discoverability, versioning, and reliability compared to Google Drive links. Args: file_id (str): Not directly used for HuggingFace downloads, but kept for API compatibility. Can be used as a fallback for repo_id if repo_id is not provided. path_local (str): The file is downloaded to this local path. repo_id (str, optional): HuggingFace Hub repository ID in the format 'username/repo_name'. If not provided, attempts to parse from file_id. filename (str, optional): Name of the file to download from the repository. If not provided, uses the basename from path_local."
},
{
"ref":"facetorch.downloader.DownloaderHuggingFace.run",
"url":5,
"doc":"Downloads a file from HuggingFace Hub. This method: 1. Creates the necessary directory structure 2. Downloads the specified file from HuggingFace Hub 3. Ensures the file is saved with the correct name at the specified path If the download fails, an informative error message is printed.",
"func":1
},
{
"ref":"facetorch.logger",
"url":6,
"doc":""
},
{
"ref":"facetorch.logger.CustomJsonFormatter",
"url":6,
"doc":"A custom formatter to format logging records as json strings. Extra values will be formatted as str() if not supported by json default encoder :param json_default: a function for encoding non-standard objects as outlined in https: docs.python.org/3/library/json.html :param json_encoder: optional custom encoder :param json_serializer: a :meth: json.dumps -compatible callable that will be used to serialize the log record. :param json_indent: an optional :meth: json.dumps -compatible numeric value that will be used to customize the indent of the output json. :param prefix: an optional string prefix added at the beginning of the formatted string :param rename_fields: an optional dict, used to rename field names in the output. Rename message to @message: {'message': '@message'} :param static_fields: an optional dict, used to add fields with static values to all logs :param json_indent: indent parameter for json.dumps :param json_ensure_ascii: ensure_ascii parameter for json.dumps :param reserved_attrs: an optional list of fields that will be skipped when outputting json log record. Defaults to all log record attributes: http: docs.python.org/library/logging.html logrecord-attributes :param timestamp: an optional string/boolean field to add a timestamp when outputting the json log record. If string is passed, timestamp will be added to log record using string as key. If True boolean is passed, timestamp key will be \"timestamp\". Defaults to False/off."
},
{
"ref":"facetorch.logger.CustomJsonFormatter.add_fields",
"url":6,
"doc":"Override this method to implement custom logic for adding fields.",
"func":1
},
{
"ref":"facetorch.logger.LoggerJsonFile",
"url":6,
"doc":"Logger in json format that writes to a file and console. Args: name (str): Name of the logger. level (str): Level of the logger. path_file (str): Path to the log file. json_format (str): Format of the log record. Attributes: logger (logging.Logger): Logger object."
},
{
"ref":"facetorch.logger.LoggerJsonFile.configure",
"url":6,
"doc":"Configures the logger.",
"func":1
},
{
"ref":"facetorch.analyzer",
"url":7,
"doc":""
},
{
"ref":"facetorch.analyzer.FaceAnalyzer",
"url":7,
"doc":"FaceAnalyzer is the main class that reads images, runs face detection, tensor unification and facial feature prediction. It also draws bounding boxes and facial landmarks over the image. The following components are used: 1. Reader - reads the image and returns an ImageData object containing the image tensor. 2. Detector - wrapper around a neural network that detects faces. 3. Unifier - processor that unifies sizes of all faces and normalizes them between 0 and 1. 4. Predictor dict - dict of wrappers around neural networks trained to analyze facial features. 5. Utilizer dict - dict of utilizer processors that can for example extract 3D face landmarks or draw boxes over the image. Args: cfg (OmegaConf): Config object with image reader, face detector, unifier and predictor configurations. Attributes: cfg (OmegaConf): Config object with image reader, face detector, unifier and predictor configurations. reader (BaseReader): Reader object that reads the image and returns an ImageData object containing the image tensor. detector (FaceDetector): FaceDetector object that wraps a neural network that detects faces. unifier (FaceUnifier): FaceUnifier object that unifies sizes of all faces and normalizes them between 0 and 1. predictors (Dict[str, FacePredictor]): Dict of FacePredictor objects that predict facial features. Key is the name of the predictor. utilizers (Dict[str, FaceUtilizer]): Dict of FaceUtilizer objects that can extract 3D face landmarks, draw boxes over the image, etc. Key is the name of the utilizer. logger (logging.Logger): Logger object that logs messages to the console or to a file."
},
{
"ref":"facetorch.analyzer.FaceAnalyzer.run",
"url":7,
"doc":"Reads image, detects faces, unifies the detected faces, predicts facial features and returns analyzed data. Args: image_source (Optional[Union[str, torch.Tensor, np.ndarray, bytes, Image.Image ): Input to be analyzed. If None, path_image or tensor must be provided. Default: None. path_image (Optional[str]): Path to the image to be analyzed. If None, tensor must be provided. Default: None. batch_size (int): Batch size for making predictions on the faces. Default is 8. fix_img_size (bool): If True, resizes the image to the size specified in reader. Default is False. return_img_data (bool): If True, returns all image data including tensors, otherwise only returns the faces. Default is False. include_tensors (bool): If True, removes tensors from the returned data object. Default is False. path_output (Optional[str]): Path where to save the image with detected faces. If None, the image is not saved. Default: None. tensor (Optional[torch.Tensor]): Image tensor to be analyzed. If None, path_image must be provided. Default: None. Returns: Union[Response, ImageData]: If return_img_data is False, returns a Response object containing the faces and their facial features. If return_img_data is True, returns the entire ImageData object.",
"func":1
},
{
"ref":"facetorch.analyzer.detector",
"url":8,
"doc":""
},
{
"ref":"facetorch.analyzer.detector.FaceDetector",
"url":8,
"doc":"FaceDetector is a wrapper around a neural network model that is trained to detect faces. Args: downloader (BaseDownloader): Downloader that downloads the model. device (torch.device): Torch device cpu or cuda for the model. preprocessor (BaseDetPreProcessor): Preprocessor that runs before the model. postprocessor (BaseDetPostProcessor): Postprocessor that runs after the model."
},
{
"ref":"facetorch.analyzer.detector.FaceDetector.run",
"url":8,
"doc":"Detect all faces in the image. Args: ImageData: ImageData object containing the image tensor with values between 0 - 255 and shape (batch_size, channels, height, width). Returns: ImageData: Image data object with Detection tensors and detected Face objects.",
"func":1
},
{
"ref":"facetorch.analyzer.detector.FaceDetector.load_model",
"url":1,
"doc":"Loads the TorchScript model. Returns: Union[torch.jit.ScriptModule, torch.jit.TracedModule]: Loaded TorchScript model.",
"func":1
},
{
"ref":"facetorch.analyzer.detector.FaceDetector.inference",
"url":1,
"doc":"Inference the model with the given tensor. Args: tensor (torch.Tensor): Input tensor for the model. Returns: Union[torch.Tensor, Tuple[torch.Tensor : Output tensor or tuple of tensors.",
"func":1
},
{
"ref":"facetorch.analyzer.detector.post",
"url":9,
"doc":""
},
{
"ref":"facetorch.analyzer.detector.post.BaseDetPostProcessor",
"url":9,
"doc":"Base class for detector post processors. All detector post processors should subclass it. All subclass should overwrite: - Methods: run , used for running the processing Args: device (torch.device): Torch device cpu or cuda. transform (transforms.Compose): Transform compose object to be applied to the image. optimize_transform (bool): Whether to optimize the transform."
},
{
"ref":"facetorch.analyzer.detector.post.BaseDetPostProcessor.run",
"url":9,
"doc":"Abstract method that runs the detector post processing functionality and returns the data object. Args: data (ImageData): ImageData object containing the image tensor. logits (Union[torch.Tensor, Tuple[torch.Tensor ): Output of the detector model. Returns: ImageData: Image data object with Detection tensors and detected Face objects.",
"func":1
},
{
"ref":"facetorch.analyzer.detector.post.BaseDetPostProcessor.optimize",
"url":1,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.detector.post.PriorBox",
"url":9,
"doc":"PriorBox class for generating prior boxes. Args: min_sizes (List[List[int ): List of list of minimum sizes for each feature map. steps (List[int]): List of steps for each feature map. clip (bool): Whether to clip the prior boxes to the image boundaries."
},
{
"ref":"facetorch.analyzer.detector.post.PriorBox.forward",
"url":9,
"doc":"Generate prior boxes for each feature map. Args: dims (Dimensions): Dimensions of the image. Returns: torch.Tensor: Tensor of prior boxes.",
"func":1
},
{
"ref":"facetorch.analyzer.detector.post.PostRetFace",
"url":9,
"doc":"Initialize the detector postprocessor. Modified from https: github.com/biubug6/Pytorch_Retinaface. Args: transform (Compose): Composed Torch transform object. device (torch.device): Torch device cpu or cuda. optimize_transform (bool): Whether to optimize the transform. confidence_threshold (float): Confidence threshold for face detection. top_k (int): Top K faces to keep before NMS. nms_threshold (float): NMS threshold. keep_top_k (int): Keep top K faces after NMS. score_threshold (float): Score threshold for face detection. prior_box (PriorBox): PriorBox object. variance (List[float]): Prior box variance. reverse_colors (bool): Whether to reverse the colors of the image tensor from RGB to BGR or vice versa. If False, the colors remain unchanged. Default: False. expand_box_ratio (float): Expand the box by this ratio. Default: 0.0."
},
{
"ref":"facetorch.analyzer.detector.post.PostRetFace.run",
"url":9,
"doc":"Run the detector postprocessor. Args: data (ImageData): ImageData object containing the image tensor. logits (Union[torch.Tensor, Tuple[torch.Tensor ): Output of the detector model. Returns: ImageData: Image data object with detection tensors and detected Face objects.",
"func":1
},
{
"ref":"facetorch.analyzer.detector.post.PostRetFace.optimize",
"url":1,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.detector.pre",
"url":10,
"doc":""
},
{
"ref":"facetorch.analyzer.detector.pre.BaseDetPreProcessor",
"url":10,
"doc":"Base class for detector pre processors. All detector pre processors should subclass it. All subclass should overwrite: - Methods: run , used for running the processing Args: device (torch.device): Torch device cpu or cuda. transform (transforms.Compose): Transform compose object to be applied to the image. optimize_transform (bool): Whether to optimize the transform."
},
{
"ref":"facetorch.analyzer.detector.pre.BaseDetPreProcessor.run",
"url":10,
"doc":"Abstract method that runs the detector pre processing functionality. Returns a batch of preprocessed face tensors. Args: data (ImageData): ImageData object containing the image tensor. Returns: ImageData: ImageData object containing the image tensor preprocessed for the detector.",
"func":1
},
{
"ref":"facetorch.analyzer.detector.pre.BaseDetPreProcessor.optimize",
"url":1,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.detector.pre.DetectorPreProcessor",
"url":10,
"doc":"Initialize the detector preprocessor. Args: transform (Compose): Composed Torch transform object. device (torch.device): Torch device cpu or cuda. optimize_transform (bool): Whether to optimize the transform. reverse_colors (bool): Whether to reverse the colors of the image tensor from RGB to BGR or vice versa. If False, the colors remain unchanged."
},
{
"ref":"facetorch.analyzer.detector.pre.DetectorPreProcessor.run",
"url":10,
"doc":"Run the detector preprocessor on the image tensor in BGR format and return the transformed image tensor. Args: data (ImageData): ImageData object containing the image tensor. Returns: ImageData: ImageData object containing the preprocessed image tensor.",
"func":1
},
{
"ref":"facetorch.analyzer.detector.pre.DetectorPreProcessor.optimize",
"url":1,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.detector.core",
"url":11,
"doc":""
},
{
"ref":"facetorch.analyzer.detector.core.FaceDetector",
"url":11,
"doc":"FaceDetector is a wrapper around a neural network model that is trained to detect faces. Args: downloader (BaseDownloader): Downloader that downloads the model. device (torch.device): Torch device cpu or cuda for the model. preprocessor (BaseDetPreProcessor): Preprocessor that runs before the model. postprocessor (BaseDetPostProcessor): Postprocessor that runs after the model."
},
{
"ref":"facetorch.analyzer.detector.core.FaceDetector.run",
"url":11,
"doc":"Detect all faces in the image. Args: ImageData: ImageData object containing the image tensor with values between 0 - 255 and shape (batch_size, channels, height, width). Returns: ImageData: Image data object with Detection tensors and detected Face objects.",
"func":1
},
{
"ref":"facetorch.analyzer.detector.core.FaceDetector.load_model",
"url":1,
"doc":"Loads the TorchScript model. Returns: Union[torch.jit.ScriptModule, torch.jit.TracedModule]: Loaded TorchScript model.",
"func":1
},
{
"ref":"facetorch.analyzer.detector.core.FaceDetector.inference",
"url":1,
"doc":"Inference the model with the given tensor. Args: tensor (torch.Tensor): Input tensor for the model. Returns: Union[torch.Tensor, Tuple[torch.Tensor : Output tensor or tuple of tensors.",
"func":1
},
{
"ref":"facetorch.analyzer.unifier",
"url":12,
"doc":""
},
{
"ref":"facetorch.analyzer.unifier.FaceUnifier",
"url":12,
"doc":"FaceUnifier is a transform based processor that can unify sizes of all faces and normalize them between 0 and 1. Args: transform (Compose): Composed Torch transform object. device (torch.device): Torch device cpu or cuda object. optimize_transform (bool): Whether to optimize the transform."
},
{
"ref":"facetorch.analyzer.unifier.FaceUnifier.run",
"url":12,
"doc":"Runs unifying transform on each face tensor one by one. Args: data (ImageData): ImageData object containing the face tensors. Returns: ImageData: ImageData object containing the unified face tensors normalized between 0 and 1.",
"func":1
},
{
"ref":"facetorch.analyzer.unifier.FaceUnifier.optimize",
"url":1,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.unifier.core",
"url":13,
"doc":""
},
{
"ref":"facetorch.analyzer.unifier.core.FaceUnifier",
"url":13,
"doc":"FaceUnifier is a transform based processor that can unify sizes of all faces and normalize them between 0 and 1. Args: transform (Compose): Composed Torch transform object. device (torch.device): Torch device cpu or cuda object. optimize_transform (bool): Whether to optimize the transform."
},
{
"ref":"facetorch.analyzer.unifier.core.FaceUnifier.run",
"url":13,
"doc":"Runs unifying transform on each face tensor one by one. Args: data (ImageData): ImageData object containing the face tensors. Returns: ImageData: ImageData object containing the unified face tensors normalized between 0 and 1.",
"func":1
},
{
"ref":"facetorch.analyzer.unifier.core.FaceUnifier.optimize",
"url":1,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor",
"url":14,
"doc":""
},
{
"ref":"facetorch.analyzer.predictor.FacePredictor",
"url":14,
"doc":"FacePredictor is a wrapper around a neural network model that is trained to predict facial features. Args: downloader (BaseDownloader): Downloader that downloads the model. device (torch.device): Torch device cpu or cuda for the model. preprocessor (BasePredPostProcessor): Preprocessor that runs before the model. postprocessor (BasePredPostProcessor): Postprocessor that runs after the model."
},
{
"ref":"facetorch.analyzer.predictor.FacePredictor.run",
"url":14,
"doc":"Predicts facial features. Args: faces (torch.Tensor): Torch tensor containing a batch of faces with values between 0-1 and shape (batch_size, channels, height, width). Returns: (List[Prediction]): List of Prediction data objects. One for each face in the batch.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.FacePredictor.load_model",
"url":1,
"doc":"Loads the TorchScript model. Returns: Union[torch.jit.ScriptModule, torch.jit.TracedModule]: Loaded TorchScript model.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.FacePredictor.inference",
"url":1,
"doc":"Inference the model with the given tensor. Args: tensor (torch.Tensor): Input tensor for the model. Returns: Union[torch.Tensor, Tuple[torch.Tensor : Output tensor or tuple of tensors.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.post",
"url":15,
"doc":""
},
{
"ref":"facetorch.analyzer.predictor.post.BasePredPostProcessor",
"url":15,
"doc":"Base class for predictor post processors. All predictor post processors should subclass it. All subclass should overwrite: - Methods: run , used for running the processing Args: device (torch.device): Torch device cpu or cuda. transform (transforms.Compose): Transform compose object to be applied to the image. optimize_transform (bool): Whether to optimize the transform. labels (List[str]): List of labels."
},
{
"ref":"facetorch.analyzer.predictor.post.BasePredPostProcessor.create_pred_list",
"url":15,
"doc":"Create a list of predictions. Args: preds (torch.Tensor): Tensor of predictions, shape (batch, _). indices (List[int]): List of label indices, one for each sample. Returns: List[Prediction]: List of predictions.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.post.BasePredPostProcessor.run",
"url":15,
"doc":"Abstract method that runs the predictor post processing functionality and returns a list of prediction data structures, one for each face in the batch. Args: preds (Union[torch.Tensor, Tuple[torch.Tensor ): Output of the predictor model. Returns: List[Prediction]: List of predictions.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.post.BasePredPostProcessor.optimize",
"url":1,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.post.PostArgMax",
"url":15,
"doc":"Initialize the predictor postprocessor that runs argmax on the prediction tensor and returns a list of prediction data structures. Args: transform (Compose): Composed Torch transform object. device (torch.device): Torch device cpu or cuda. optimize_transform (bool): Whether to optimize the transform using TorchScript. labels (List[str]): List of labels. dim (int): Axis along which to apply the argmax."
},
{
"ref":"facetorch.analyzer.predictor.post.PostArgMax.run",
"url":15,
"doc":"Post-processes the prediction tensor using argmax and returns a list of prediction data structures, one for each face. Args: preds (torch.Tensor): Batch prediction tensor. Returns: List[Prediction]: List of prediction data structures containing the predicted labels and confidence scores for each face in the batch.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.post.PostArgMax.create_pred_list",
"url":15,
"doc":"Create a list of predictions. Args: preds (torch.Tensor): Tensor of predictions, shape (batch, _). indices (List[int]): List of label indices, one for each sample. Returns: List[Prediction]: List of predictions.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.post.PostArgMax.optimize",
"url":1,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.post.PostSigmoidBinary",
"url":15,
"doc":"Initialize the predictor postprocessor that runs sigmoid on the prediction tensor and returns a list of prediction data structures. Args: transform (Compose): Composed Torch transform object. device (torch.device): Torch device cpu or cuda. optimize_transform (bool): Whether to optimize the transform using TorchScript. labels (List[str]): List of labels. threshold (float): Probability threshold for positive class."
},
{
"ref":"facetorch.analyzer.predictor.post.PostSigmoidBinary.run",
"url":15,
"doc":"Post-processes the prediction tensor using argmax and returns a list of prediction data structures, one for each face. Args: preds (torch.Tensor): Batch prediction tensor. Returns: List[Prediction]: List of prediction data structures containing the predicted labelsand confidence scores for each face in the batch.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.post.PostSigmoidBinary.create_pred_list",
"url":15,
"doc":"Create a list of predictions. Args: preds (torch.Tensor): Tensor of predictions, shape (batch, _). indices (List[int]): List of label indices, one for each sample. Returns: List[Prediction]: List of predictions.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.post.PostSigmoidBinary.optimize",
"url":1,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.post.PostEmbedder",
"url":15,
"doc":"Initialize the predictor postprocessor that extracts the embedding from the prediction tensor and returns a list of prediction data structures. Args: transform (Compose): Composed Torch transform object. device (torch.device): Torch device cpu or cuda. optimize_transform (bool): Whether to optimize the transform using TorchScript. labels (List[str]): List of labels."
},
{
"ref":"facetorch.analyzer.predictor.post.PostEmbedder.run",
"url":15,
"doc":"Extracts the embedding from the prediction tensor and returns a list of prediction data structures, one for each face. Args: preds (torch.Tensor): Batch prediction tensor. Returns: List[Prediction]: List of prediction data structures containing the predicted embeddings.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.post.PostEmbedder.create_pred_list",
"url":15,
"doc":"Create a list of predictions. Args: preds (torch.Tensor): Tensor of predictions, shape (batch, _). indices (List[int]): List of label indices, one for each sample. Returns: List[Prediction]: List of predictions.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.post.PostEmbedder.optimize",
"url":1,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.post.PostMultiLabel",
"url":15,
"doc":"Initialize the predictor postprocessor that extracts multiple labels from the confidence scores. Args: transform (Compose): Composed Torch transform object. device (torch.device): Torch device cpu or cuda. optimize_transform (bool): Whether to optimize the transform using TorchScript. labels (List[str]): List of labels. dim (int): Axis along which to apply the softmax. threshold (float): Probability threshold for including a label. Only labels with a confidence score above the threshold are included. Defaults to 0.5."
},
{
"ref":"facetorch.analyzer.predictor.post.PostMultiLabel.run",
"url":15,
"doc":"Extracts multiple labels and puts them in other[multi] predictions. The most likely label is put in the label field. Confidence scores are returned in the logits field. Args: preds (torch.Tensor): Batch prediction tensor. Returns: List[Prediction]: List of prediction data structures containing the most prevailing label, confidence scores, and multiple labels for each face.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.post.PostMultiLabel.create_pred_list",
"url":15,
"doc":"Create a list of predictions. Args: preds (torch.Tensor): Tensor of predictions, shape (batch, _). indices (List[int]): List of label indices, one for each sample. Returns: List[Prediction]: List of predictions.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.post.PostMultiLabel.optimize",
"url":1,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.post.PostLabelConfidencePairs",
"url":15,
"doc":"Initialize the predictor postprocessor that zips the confidence scores with the labels. Args: transform (Compose): Composed Torch transform object. device (torch.device): Torch device cpu or cuda. optimize_transform (bool): Whether to optimize the transform using TorchScript. labels (List[str]): List of labels. offsets (Optional[List[float , optional): List of offsets to add to the confidence scores. Defaults to None."
},
{
"ref":"facetorch.analyzer.predictor.post.PostLabelConfidencePairs.run",
"url":15,
"doc":"Extracts the confidence scores and puts them in other[label] predictions. Args: preds (torch.Tensor): Batch prediction tensor. Returns: List[Prediction]: List of prediction data structures containing the logits and label logit pairs.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.post.PostLabelConfidencePairs.create_pred_list",
"url":15,
"doc":"Create a list of predictions. Args: preds (torch.Tensor): Tensor of predictions, shape (batch, _). indices (List[int]): List of label indices, one for each sample. Returns: List[Prediction]: List of predictions.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.post.PostLabelConfidencePairs.optimize",
"url":1,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.pre",
"url":16,
"doc":""
},
{
"ref":"facetorch.analyzer.predictor.pre.BasePredPreProcessor",
"url":16,
"doc":"Base class for predictor pre processors. All predictor pre processors should subclass it. All subclass should overwrite: - Methods: run , used for running the processing Args: device (torch.device): Torch device cpu or cuda. transform (transforms.Compose): Transform compose object to be applied to the image. optimize_transform (bool): Whether to optimize the transform."
},
{
"ref":"facetorch.analyzer.predictor.pre.BasePredPreProcessor.run",
"url":16,
"doc":"Abstract method that runs the predictor pre processing functionality and returns a batch of preprocessed face tensors. Args: faces (torch.Tensor): Batch of face tensors with shape (batch, channels, height, width). Returns: torch.Tensor: Batch of preprocessed face tensors with shape (batch, channels, height, width).",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.pre.BasePredPreProcessor.optimize",
"url":1,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.pre.PredictorPreProcessor",
"url":16,
"doc":"Torch transform based pre-processor that is applied to face tensors before they are passed to the predictor model. Args: transform (transforms.Compose): Composed Torch transform object. device (torch.device): Torch device cpu or cuda. optimize_transform (bool): Whether to optimize the transform. reverse_colors (bool): Whether to reverse the colors of the image tensor"
},
{
"ref":"facetorch.analyzer.predictor.pre.PredictorPreProcessor.run",
"url":16,
"doc":"Runs the trasform on a batch of face tensors. Args: faces (torch.Tensor): Batch of face tensors. Returns: torch.Tensor: Batch of preprocessed face tensors.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.pre.PredictorPreProcessor.optimize",
"url":1,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.core",
"url":17,
"doc":""
},
{
"ref":"facetorch.analyzer.predictor.core.FacePredictor",
"url":17,
"doc":"FacePredictor is a wrapper around a neural network model that is trained to predict facial features. Args: downloader (BaseDownloader): Downloader that downloads the model. device (torch.device): Torch device cpu or cuda for the model. preprocessor (BasePredPostProcessor): Preprocessor that runs before the model. postprocessor (BasePredPostProcessor): Postprocessor that runs after the model."
},
{
"ref":"facetorch.analyzer.predictor.core.FacePredictor.run",
"url":17,
"doc":"Predicts facial features. Args: faces (torch.Tensor): Torch tensor containing a batch of faces with values between 0-1 and shape (batch_size, channels, height, width). Returns: (List[Prediction]): List of Prediction data objects. One for each face in the batch.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.core.FacePredictor.load_model",
"url":1,
"doc":"Loads the TorchScript model. Returns: Union[torch.jit.ScriptModule, torch.jit.TracedModule]: Loaded TorchScript model.",
"func":1
},
{
"ref":"facetorch.analyzer.predictor.core.FacePredictor.inference",
"url":1,
"doc":"Inference the model with the given tensor. Args: tensor (torch.Tensor): Input tensor for the model. Returns: Union[torch.Tensor, Tuple[torch.Tensor : Output tensor or tuple of tensors.",
"func":1
},
{
"ref":"facetorch.analyzer.utilizer",
"url":18,
"doc":""
},
{
"ref":"facetorch.analyzer.utilizer.Lmk3DMeshPose",
"url":18,
"doc":"Initializes the Lmk3DMeshPose class. This class is used to convert the face parameter vector to 3D landmarks, mesh and pose. Args: transform (Compose): Composed Torch transform object. device (torch.device): Torch device cpu or cuda object. optimize_transform (bool): Whether to optimize the transform. downloader_meta (BaseDownloader): Downloader for metadata. image_size (int): Standard size of the face image."
},
{
"ref":"facetorch.analyzer.utilizer.Lmk3DMeshPose.run",
"url":18,
"doc":"Runs the Lmk3DMeshPose class functionality - convert the face parameter vector to 3D landmarks, mesh and pose. Adds the following attributes to the data object: - landmark  y, x, z], 68 (points)] - mesh  y, x, z], 53215 (points)] - pose (Euler angles [yaw, pitch, roll] and translation [y, x, z]) Args: data (ImageData): ImageData object containing most of the data including the predictions. Returns: ImageData: ImageData object containing lmk3d, mesh and pose.",
"func":1
},
{
"ref":"facetorch.analyzer.utilizer.Lmk3DMeshPose.optimize",
"url":1,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.utilizer.BoxDrawer",
"url":18,
"doc":"Initializes the BoxDrawer class. This class is used to draw the face boxes to the image tensor. Args: transform (Compose): Composed Torch transform object. device (torch.device): Torch device cpu or cuda object. optimize_transform (bool): Whether to optimize the transform. color (str): Color of the boxes. line_width (int): Line width of the boxes."
},
{
"ref":"facetorch.analyzer.utilizer.BoxDrawer.run",
"url":18,
"doc":"Draws face boxes to the image tensor. Args: data (ImageData): ImageData object containing the image tensor and face locations. Returns: ImageData: ImageData object containing the image tensor with face boxes.",
"func":1
},
{
"ref":"facetorch.analyzer.utilizer.BoxDrawer.optimize",
"url":1,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.utilizer.LandmarkDrawerTorch",
"url":18,
"doc":"Initializes the LandmarkDrawer class. This class is used to draw the 3D face landmarks to the image tensor. Args: transform (Compose): Composed Torch transform object. device (torch.device): Torch device cpu or cuda object. optimize_transform (bool): Whether to optimize the transform. width (int): Marker keypoint width. color (str): Marker color."
},
{
"ref":"facetorch.analyzer.utilizer.LandmarkDrawerTorch.run",
"url":18,
"doc":"Draws 3D face landmarks to the image tensor. Args: data (ImageData): ImageData object containing the image tensor and 3D face landmarks. Returns: ImageData: ImageData object containing the image tensor with 3D face landmarks.",
"func":1
},
{
"ref":"facetorch.analyzer.utilizer.LandmarkDrawerTorch.optimize",
"url":1,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.utilizer.ImageSaver",
"url":18,
"doc":"Initializes the ImageSaver class. This class is used to save the image tensor to an image file. Args: transform (Compose): Composed Torch transform object. device (torch.device): Torch device cpu or cuda object. optimize_transform (bool): Whether to optimize the transform."
},
{
"ref":"facetorch.analyzer.utilizer.ImageSaver.run",
"url":18,
"doc":"Saves the image tensor to an image file, if the path_output attribute of ImageData is not None. Args: data (ImageData): ImageData object containing the img tensor. Returns: ImageData: ImageData object containing the same data as the input.",
"func":1
},
{
"ref":"facetorch.analyzer.utilizer.ImageSaver.optimize",
"url":1,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.utilizer.save",
"url":19,
"doc":""
},
{
"ref":"facetorch.analyzer.utilizer.save.ImageSaver",
"url":19,
"doc":"Initializes the ImageSaver class. This class is used to save the image tensor to an image file. Args: transform (Compose): Composed Torch transform object. device (torch.device): Torch device cpu or cuda object. optimize_transform (bool): Whether to optimize the transform."
},
{
"ref":"facetorch.analyzer.utilizer.save.ImageSaver.run",
"url":19,
"doc":"Saves the image tensor to an image file, if the path_output attribute of ImageData is not None. Args: data (ImageData): ImageData object containing the img tensor. Returns: ImageData: ImageData object containing the same data as the input.",
"func":1
},
{
"ref":"facetorch.analyzer.utilizer.save.ImageSaver.optimize",
"url":1,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.utilizer.draw",
"url":20,
"doc":""
},
{
"ref":"facetorch.analyzer.utilizer.draw.BoxDrawer",
"url":20,
"doc":"Initializes the BoxDrawer class. This class is used to draw the face boxes to the image tensor. Args: transform (Compose): Composed Torch transform object. device (torch.device): Torch device cpu or cuda object. optimize_transform (bool): Whether to optimize the transform. color (str): Color of the boxes. line_width (int): Line width of the boxes."
},
{
"ref":"facetorch.analyzer.utilizer.draw.BoxDrawer.run",
"url":20,
"doc":"Draws face boxes to the image tensor. Args: data (ImageData): ImageData object containing the image tensor and face locations. Returns: ImageData: ImageData object containing the image tensor with face boxes.",
"func":1
},
{
"ref":"facetorch.analyzer.utilizer.draw.BoxDrawer.optimize",
"url":1,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.utilizer.draw.LandmarkDrawerTorch",
"url":20,
"doc":"Initializes the LandmarkDrawer class. This class is used to draw the 3D face landmarks to the image tensor. Args: transform (Compose): Composed Torch transform object. device (torch.device): Torch device cpu or cuda object. optimize_transform (bool): Whether to optimize the transform. width (int): Marker keypoint width. color (str): Marker color."
},
{
"ref":"facetorch.analyzer.utilizer.draw.LandmarkDrawerTorch.run",
"url":20,
"doc":"Draws 3D face landmarks to the image tensor. Args: data (ImageData): ImageData object containing the image tensor and 3D face landmarks. Returns: ImageData: ImageData object containing the image tensor with 3D face landmarks.",
"func":1
},
{
"ref":"facetorch.analyzer.utilizer.draw.LandmarkDrawerTorch.optimize",
"url":1,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.utilizer.align",
"url":21,
"doc":""
},
{
"ref":"facetorch.analyzer.utilizer.align.Lmk3DMeshPose",
"url":21,
"doc":"Initializes the Lmk3DMeshPose class. This class is used to convert the face parameter vector to 3D landmarks, mesh and pose. Args: transform (Compose): Composed Torch transform object. device (torch.device): Torch device cpu or cuda object. optimize_transform (bool): Whether to optimize the transform. downloader_meta (BaseDownloader): Downloader for metadata. image_size (int): Standard size of the face image."
},
{
"ref":"facetorch.analyzer.utilizer.align.Lmk3DMeshPose.run",
"url":21,
"doc":"Runs the Lmk3DMeshPose class functionality - convert the face parameter vector to 3D landmarks, mesh and pose. Adds the following attributes to the data object: - landmark  y, x, z], 68 (points)] - mesh  y, x, z], 53215 (points)] - pose (Euler angles [yaw, pitch, roll] and translation [y, x, z]) Args: data (ImageData): ImageData object containing most of the data including the predictions. Returns: ImageData: ImageData object containing lmk3d, mesh and pose.",
"func":1
},
{
"ref":"facetorch.analyzer.utilizer.align.Lmk3DMeshPose.optimize",
"url":1,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.reader",
"url":22,
"doc":""
},
{
"ref":"facetorch.analyzer.reader.ImageReader",
"url":22,
"doc":"ImageReader is a wrapper around a functionality for reading images by Torchvision. Args: transform (torchvision.transforms.Compose): Transform compose object to be applied to the image, if fix_image_size is True. device (torch.device): Torch device cpu or cuda object. optimize_transform (bool): Whether to optimize the transforms that are: resizing the image to a fixed size."
},
{
"ref":"facetorch.analyzer.reader.ImageReader.run",
"url":22,
"doc":"Reads an image from a path and returns a tensor of the image with values between 0-255 and shape (batch, channels, height, width). The order of color channels is RGB. PyTorch and Torchvision are used to read the image. Args: path_image (str): Path to the image. fix_img_size (bool): Whether to resize the image to a fixed size. If False, the size_portrait and size_landscape are ignored. Default is False. Returns: ImageData: ImageData object with image tensor and pil Image.",
"func":1
},
{
"ref":"facetorch.analyzer.reader.ImageReader.process_tensor",
"url":1,
"doc":"Read a tensor and return a data object containing a tensor of the image with shape (batch, channels, height, width). Args: tensor (torch.Tensor): Tensor of a single image with RGB values between 0-255 and shape (channels, height, width). fix_img_size (bool): Whether to resize the image to a fixed size. If False, the size_portrait and size_landscape are ignored. Default is False.",
"func":1
},
{
"ref":"facetorch.analyzer.reader.ImageReader.optimize",
"url":1,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.reader.TensorReader",
"url":22,
"doc":"TensorReader is a wrapper around a functionality for reading tensors by Torchvision. Args: transform (torchvision.transforms.Compose): Transform compose object to be applied to the image, if fix_image_size is True. device (torch.device): Torch device cpu or cuda object. optimize_transform (bool): Whether to optimize the transforms that are: resizing the image to a fixed size."
},
{
"ref":"facetorch.analyzer.reader.TensorReader.run",
"url":22,
"doc":"Reads a tensor and returns a tensor of the image with values between 0-255 and shape (batch, channels, height, width). The order of color channels is RGB. PyTorch and Torchvision are used to read the image. Args: tensor (torch.Tensor): Tensor of a single image with RGB values between 0-255 and shape (channels, height, width). fix_img_size (bool): Whether to resize the image to a fixed size. If False, the size_portrait and size_landscape are ignored. Default is False. Returns: ImageData: ImageData object with image tensor and pil Image.",
"func":1
},
{
"ref":"facetorch.analyzer.reader.TensorReader.process_tensor",
"url":1,
"doc":"Read a tensor and return a data object containing a tensor of the image with shape (batch, channels, height, width). Args: tensor (torch.Tensor): Tensor of a single image with RGB values between 0-255 and shape (channels, height, width). fix_img_size (bool): Whether to resize the image to a fixed size. If False, the size_portrait and size_landscape are ignored. Default is False.",
"func":1
},
{
"ref":"facetorch.analyzer.reader.TensorReader.optimize",
"url":1,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.reader.UniversalReader",
"url":22,
"doc":"UniversalReader can read images from a path, URL, tensor, numpy array, bytes or PIL Image and return an ImageData object containing the image tensor. Args: transform (torchvision.transforms.Compose): Transform compose object to be applied to the image, if fix_image_size is True. device (torch.device): Torch device cpu or cuda object. optimize_transform (bool): Whether to optimize the transforms that are: resizing the image to a fixed size."
},
{
"ref":"facetorch.analyzer.reader.UniversalReader.run",
"url":22,
"doc":"Reads an image from a path, URL, tensor, numpy array, bytes or PIL Image and returns a tensor of the image with values between 0-255 and shape (batch, channels, height, width). The order of color channels is RGB. PyTorch and Torchvision are used to read the image. Args: image_source (Union[str, torch.Tensor, np.ndarray, bytes, Image.Image]): Image source to be read. fix_img_size (bool): Whether to resize the image to a fixed size. If False, the size_portrait and size_landscape are ignored. Default is False. Returns: ImageData: ImageData object with image tensor and pil Image.",
"func":1
},
{
"ref":"facetorch.analyzer.reader.UniversalReader.read_tensor",
"url":22,
"doc":"",
"func":1
},
{
"ref":"facetorch.analyzer.reader.UniversalReader.read_pil_image",
"url":22,
"doc":"",
"func":1
},
{
"ref":"facetorch.analyzer.reader.UniversalReader.read_numpy_array",
"url":22,
"doc":"",
"func":1
},
{
"ref":"facetorch.analyzer.reader.UniversalReader.read_image_from_bytes",
"url":22,
"doc":"",
"func":1
},
{
"ref":"facetorch.analyzer.reader.UniversalReader.read_image_from_path",
"url":22,
"doc":"",
"func":1
},
{
"ref":"facetorch.analyzer.reader.UniversalReader.read_image_from_url",
"url":22,
"doc":"",
"func":1
},
{
"ref":"facetorch.analyzer.reader.UniversalReader.process_tensor",
"url":1,
"doc":"Read a tensor and return a data object containing a tensor of the image with shape (batch, channels, height, width). Args: tensor (torch.Tensor): Tensor of a single image with RGB values between 0-255 and shape (channels, height, width). fix_img_size (bool): Whether to resize the image to a fixed size. If False, the size_portrait and size_landscape are ignored. Default is False.",
"func":1
},
{
"ref":"facetorch.analyzer.reader.UniversalReader.optimize",
"url":1,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.reader.core",
"url":23,
"doc":""
},
{
"ref":"facetorch.analyzer.reader.core.UniversalReader",
"url":23,
"doc":"UniversalReader can read images from a path, URL, tensor, numpy array, bytes or PIL Image and return an ImageData object containing the image tensor. Args: transform (torchvision.transforms.Compose): Transform compose object to be applied to the image, if fix_image_size is True. device (torch.device): Torch device cpu or cuda object. optimize_transform (bool): Whether to optimize the transforms that are: resizing the image to a fixed size."
},
{
"ref":"facetorch.analyzer.reader.core.UniversalReader.run",
"url":23,
"doc":"Reads an image from a path, URL, tensor, numpy array, bytes or PIL Image and returns a tensor of the image with values between 0-255 and shape (batch, channels, height, width). The order of color channels is RGB. PyTorch and Torchvision are used to read the image. Args: image_source (Union[str, torch.Tensor, np.ndarray, bytes, Image.Image]): Image source to be read. fix_img_size (bool): Whether to resize the image to a fixed size. If False, the size_portrait and size_landscape are ignored. Default is False. Returns: ImageData: ImageData object with image tensor and pil Image.",
"func":1
},
{
"ref":"facetorch.analyzer.reader.core.UniversalReader.read_tensor",
"url":23,
"doc":"",
"func":1
},
{
"ref":"facetorch.analyzer.reader.core.UniversalReader.read_pil_image",
"url":23,
"doc":"",
"func":1
},
{
"ref":"facetorch.analyzer.reader.core.UniversalReader.read_numpy_array",
"url":23,
"doc":"",
"func":1
},
{
"ref":"facetorch.analyzer.reader.core.UniversalReader.read_image_from_bytes",
"url":23,
"doc":"",
"func":1
},
{
"ref":"facetorch.analyzer.reader.core.UniversalReader.read_image_from_path",
"url":23,
"doc":"",
"func":1
},
{
"ref":"facetorch.analyzer.reader.core.UniversalReader.read_image_from_url",
"url":23,
"doc":"",
"func":1
},
{
"ref":"facetorch.analyzer.reader.core.UniversalReader.process_tensor",
"url":1,
"doc":"Read a tensor and return a data object containing a tensor of the image with shape (batch, channels, height, width). Args: tensor (torch.Tensor): Tensor of a single image with RGB values between 0-255 and shape (channels, height, width). fix_img_size (bool): Whether to resize the image to a fixed size. If False, the size_portrait and size_landscape are ignored. Default is False.",
"func":1
},
{
"ref":"facetorch.analyzer.reader.core.UniversalReader.optimize",
"url":1,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.reader.core.ImageReader",
"url":23,
"doc":"ImageReader is a wrapper around a functionality for reading images by Torchvision. Args: transform (torchvision.transforms.Compose): Transform compose object to be applied to the image, if fix_image_size is True. device (torch.device): Torch device cpu or cuda object. optimize_transform (bool): Whether to optimize the transforms that are: resizing the image to a fixed size."
},
{
"ref":"facetorch.analyzer.reader.core.ImageReader.run",
"url":23,
"doc":"Reads an image from a path and returns a tensor of the image with values between 0-255 and shape (batch, channels, height, width). The order of color channels is RGB. PyTorch and Torchvision are used to read the image. Args: path_image (str): Path to the image. fix_img_size (bool): Whether to resize the image to a fixed size. If False, the size_portrait and size_landscape are ignored. Default is False. Returns: ImageData: ImageData object with image tensor and pil Image.",
"func":1
},
{
"ref":"facetorch.analyzer.reader.core.ImageReader.process_tensor",
"url":1,
"doc":"Read a tensor and return a data object containing a tensor of the image with shape (batch, channels, height, width). Args: tensor (torch.Tensor): Tensor of a single image with RGB values between 0-255 and shape (channels, height, width). fix_img_size (bool): Whether to resize the image to a fixed size. If False, the size_portrait and size_landscape are ignored. Default is False.",
"func":1
},
{
"ref":"facetorch.analyzer.reader.core.ImageReader.optimize",
"url":1,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.reader.core.TensorReader",
"url":23,
"doc":"TensorReader is a wrapper around a functionality for reading tensors by Torchvision. Args: transform (torchvision.transforms.Compose): Transform compose object to be applied to the image, if fix_image_size is True. device (torch.device): Torch device cpu or cuda object. optimize_transform (bool): Whether to optimize the transforms that are: resizing the image to a fixed size."
},
{
"ref":"facetorch.analyzer.reader.core.TensorReader.run",
"url":23,
"doc":"Reads a tensor and returns a tensor of the image with values between 0-255 and shape (batch, channels, height, width). The order of color channels is RGB. PyTorch and Torchvision are used to read the image. Args: tensor (torch.Tensor): Tensor of a single image with RGB values between 0-255 and shape (channels, height, width). fix_img_size (bool): Whether to resize the image to a fixed size. If False, the size_portrait and size_landscape are ignored. Default is False. Returns: ImageData: ImageData object with image tensor and pil Image.",
"func":1
},
{
"ref":"facetorch.analyzer.reader.core.TensorReader.process_tensor",
"url":1,
"doc":"Read a tensor and return a data object containing a tensor of the image with shape (batch, channels, height, width). Args: tensor (torch.Tensor): Tensor of a single image with RGB values between 0-255 and shape (channels, height, width). fix_img_size (bool): Whether to resize the image to a fixed size. If False, the size_portrait and size_landscape are ignored. Default is False.",
"func":1
},
{
"ref":"facetorch.analyzer.reader.core.TensorReader.optimize",
"url":1,
"doc":"Optimizes the transform using torch.jit and deploys it to the device.",
"func":1
},
{
"ref":"facetorch.analyzer.core",
"url":24,
"doc":""
},
{
"ref":"facetorch.analyzer.core.FaceAnalyzer",
"url":24,
"doc":"FaceAnalyzer is the main class that reads images, runs face detection, tensor unification and facial feature prediction. It also draws bounding boxes and facial landmarks over the image. The following components are used: 1. Reader - reads the image and returns an ImageData object containing the image tensor. 2. Detector - wrapper around a neural network that detects faces. 3. Unifier - processor that unifies sizes of all faces and normalizes them between 0 and 1. 4. Predictor dict - dict of wrappers around neural networks trained to analyze facial features. 5. Utilizer dict - dict of utilizer processors that can for example extract 3D face landmarks or draw boxes over the image. Args: cfg (OmegaConf): Config object with image reader, face detector, unifier and predictor configurations. Attributes: cfg (OmegaConf): Config object with image reader, face detector, unifier and predictor configurations. reader (BaseReader): Reader object that reads the image and returns an ImageData object containing the image tensor. detector (FaceDetector): FaceDetector object that wraps a neural network that detects faces. unifier (FaceUnifier): FaceUnifier object that unifies sizes of all faces and normalizes them between 0 and 1. predictors (Dict[str, FacePredictor]): Dict of FacePredictor objects that predict facial features. Key is the name of the predictor. utilizers (Dict[str, FaceUtilizer]): Dict of FaceUtilizer objects that can extract 3D face landmarks, draw boxes over the image, etc. Key is the name of the utilizer. logger (logging.Logger): Logger object that logs messages to the console or to a file."
},
{
"ref":"facetorch.analyzer.core.FaceAnalyzer.run",
"url":24,
"doc":"Reads image, detects faces, unifies the detected faces, predicts facial features and returns analyzed data. Args: image_source (Optional[Union[str, torch.Tensor, np.ndarray, bytes, Image.Image ): Input to be analyzed. If None, path_image or tensor must be provided. Default: None. path_image (Optional[str]): Path to the image to be analyzed. If None, tensor must be provided. Default: None. batch_size (int): Batch size for making predictions on the faces. Default is 8. fix_img_size (bool): If True, resizes the image to the size specified in reader. Default is False. return_img_data (bool): If True, returns all image data including tensors, otherwise only returns the faces. Default is False. include_tensors (bool): If True, removes tensors from the returned data object. Default is False. path_output (Optional[str]): Path where to save the image with detected faces. If None, the image is not saved. Default: None. tensor (Optional[torch.Tensor]): Image tensor to be analyzed. If None, path_image must be provided. Default: None. Returns: Union[Response, ImageData]: If return_img_data is False, returns a Response object containing the faces and their facial features. If return_img_data is True, returns the entire ImageData object.",
"func":1
}
]