
import os
import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Dict, Any

class MLPipelineError(Exception):
    """Custom exception for ML pipeline failures."""
    pass

class VehicleDetectionPipeline: # it is ok, is it english leave all comments and docstrings
    """
    Orchestrates the ONNX models for full-frame vehicle detection 
    and cropped license plate detection.
    """
    
    def __init__(self, vehicle_model_path: str, plate_model_path: str, use_gpu: bool = True):
        """
        Initializes the ONNX Runtime sessions for the required models.
        
        Args:
            vehicle_model_path (str): Path to the vehicle detector .onnx file.
            plate_model_path (str): Path to the license plate detector .onnx file.
            use_gpu (bool): Whether to utilize the CUDAExecutionProvider.
        """
        self.providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        
        if not os.path.exists(vehicle_model_path) or not os.path.exists(plate_model_path):
            raise MLPipelineError("ONNX model files not found. Check the ml_models directory.")

        # Initialize ONNX Inference Sessions
        self.session_vehicle = ort.InferenceSession(vehicle_model_path, providers=self.providers)
        self.session_plate = ort.InferenceSession(plate_model_path, providers=self.providers)
        
        # Get input names dynamically from the ONNX computational graph
        self.input_name_vehicle = self.session_vehicle.get_inputs()[0].name
        self.input_name_plate = self.session_plate.get_inputs()[0].name
        
        # Map class IDs to our VehicleType schema Enum strings
        self.class_map = {
            0: bus,
            1: car,
            2: microbus,
            3: motorbike,
            4: pickup-van,
            5: truck,
            # these are the six classes of the vehicle detector from the original
            # dataset from ROBOFLOW , this one : /vehicle-detection-by9xs-4wygy/settings
        }

    def _preprocess_image(self, img: np.ndarray, target_size: int = 640) -> Tuple[np.ndarray, float]:
        """
        Resizes and normalizes the image tensor for ONNX inference.
        
        Args:
            img (np.ndarray): The raw BGR OpenCV image.
            target_size (int): The expected static input size for the YOLO model.
            
        Returns:
            Tuple[np.ndarray, float]: The NCHW formatted tensor and the scale factor used.
        """
        # Calculate scale to maintain aspect ratio while fitting into target_size
        h, w = img.shape[:2]
        scale = target_size / max(h, w)
        
        # Resize image
        img_resized = cv2.resize(img, (int(w * scale), int(h * scale)))
        
        # Create a blank square canvas (padding) and place the resized image
        canvas = np.zeros((target_size, target_size, 3), dtype=np.float32)
        canvas[:img_resized.shape[0], :img_resized.shape[1], :] = img_resized
        
        # Normalize pixel values to [0, 1] range
        canvas /= 255.0
        
        # Convert HWC (Height, Width, Channels) to NCHW (Batch, Channels, Height, Width)
        tensor = np.transpose(canvas, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        
        return tensor, scale

    def detect_vehicles(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Runs the primary YOLOv8 model to detect vehicles in the main frame.
        
        Args:
            frame (np.ndarray): The original video frame.
            conf_threshold (float): Minimum confidence probability to keep a detection.
            
        Returns:
            List[Dict[str, Any]]: List of detected vehicles with bounding boxes and classes.
        """
        tensor, scale = self._preprocess_image(frame, target_size=640)
        
        # Execute the computational graph on the GPU/CPU
        outputs = self.session_vehicle.run(None, {self.input_name_vehicle: tensor})
        predictions = outputs[0]  # Shape: (1, 4 + classes, 8400)
        
        # Transpose to (1, 8400, 4 + classes) for easier parsing
        predictions = np.squeeze(predictions).transpose()
        
        detections = []
        boxes = []
        scores = []
        class_ids = []
        
        # Parse the raw tensor
        for row in predictions:
            class_scores = row[4:]
            class_id = np.argmax(class_scores)
            max_score = class_scores[class_id]
            
            if max_score >= conf_threshold and class_id in self.class_map:
                # Extract center x, center y, width, height
                xc, yc, w, h = row[0], row[1], row[2], row[3]
                
                # Convert to x_min, y_min, x_max, y_max and reverse the scaling
                x_min = int((xc - w / 2) / scale)
                y_min = int((yc - h / 2) / scale)
                x_max = int((xc + w / 2) / scale)
                y_max = int((yc + h / 2) / scale)
                
                boxes.append([x_min, y_min, x_max, y_max])
                scores.append(float(max_score))
                class_ids.append(class_id)
                
        # Apply OpenCV's built-in NMS to filter overlapping boxes (IoU logic)
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold=0.45)
        
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append({
                    "bbox": boxes[i],
                    "score": scores[i],
                    "class": self.class_map[class_ids[i]]
                })
                
        return detections