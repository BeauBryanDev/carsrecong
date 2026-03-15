
import os
import cv2
import re
import numpy as np
import onnxruntime as ort
from typing import Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "ml_models")
OCR_MODEL_PATH = os.path.join(MODELS_DIR, "plate_ocr.onnx")

class PlateOCRProcessor:
    """
    Handles the execution of the CRNN-based recognition model exported from PaddleOCR.
    Requires a preprocessed binary/grayscale tensor of the cropped plate.
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initializes the ONNX Runtime session for the OCR model.
        
        Args:
            use_gpu (bool): Hardware acceleration flag.
        """
        self.providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        
        if not os.path.exists(OCR_MODEL_PATH):
            raise FileNotFoundError(f"OCR ONNX model not found at {OCR_MODEL_PATH}")

        #self.session = ort.InferenceSession(OCR_MODEL_PATH, providers=self.providers)
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = os.cpu_count()

        self.session = ort.InferenceSession(
            OCR_MODEL_PATH,
            sess_options=sess_options,
            providers=self.providers
        )
        
        self.input_name = self.session.get_inputs()[0].name
        
        # Standard PP-OCRv4 English dictionary (96 characters + 1 CTC blank at index 0)
        # Prevents IndexError when mapping the (Sequence, 97) Softmax tensor output.
        dict_string = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]_`~ "
        self.character_set = ['blank'] + list(dict_string)

    def _preprocess_for_crnn(self, processed_crop: np.ndarray) -> np.ndarray:
        """
        Resizes and zero-pads the image to strictly match the static ONNX computational graph 
        dimensions: [1, 3, 48, 320].
        
        Args:
            processed_crop (np.ndarray): Image processed by CLAHE and thresholding.
            
        Returns:
            np.ndarray: NCHW tensor ready for inference.
        """
        
        target_height = 48
        target_width = 320
        h, w = processed_crop.shape[:2]
        
        # Calculate resize ratio keeping the mathematical aspect ratio intact
        ratio = w / float(h)
        resized_w = int(target_height * ratio)
        
        # Clip the width mathematically to not exceed our static tensor size
        if resized_w > target_width:
            resized_w = target_width
            
        resized = cv2.resize(processed_crop, (resized_w, target_height), interpolation=cv2.INTER_AREA)
        
        # PP-OCRv4 expects a 3-channel tensor, so we convert the binary/gray image to BGR
        resized = np.repeat(resized[:, :, np.newaxis], 3, axis=2)

            
        # Create a black mathematical canvas (zero-padding) of exactly 48x320
        canvas = np.zeros((target_height, target_width, 3), dtype=np.float32)
        
        # Paste the resized plate into the left side of the canvas
        canvas[:, :resized_w, :] = resized
        
        self.canvas = np.zeros((48,320,3), dtype=np.float32)
        canvas = self.canvas
        canvas[:] = 0

        
        # Normalize to [0, 1] and transpose to NCHW format
        #tensor = canvas / 255.0
        
        tensor = canvas.astype(np.float32) * (1.0 / 255.0)

        
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        
        return tensor
        
        

    def _ctc_decode(self, predictions: np.ndarray) -> Tuple[str, float]:
        """
        Decodes the raw Softmax probabilities using CTC Greedy Decoding.
        Applies Regex filtering to guarantee only alphanumeric characters.
        
        Args:
            predictions (np.ndarray): The raw output tensor from the CRNN.
            
        Returns:
            Tuple[str, float]: The filtered decoded text and its average confidence score.
        """
        sequence = predictions[0] 
        
        raw_text = []
        confidences = []
        previous_idx = 0
        
        for step in sequence:
            char_idx = int(np.argmax(step))
            confidence = float(np.max(step))
            
            if char_idx != 0 and char_idx != previous_idx:
                raw_text.append(self.character_set[char_idx])
                confidences.append(confidence)
                
            previous_idx = char_idx
            
        unfiltered_string = "".join(raw_text)
        
        plate_pattern = r'[A-Z]{3}[0-9]{2,3}[A-Z]?'
        filtered_string = re.sub(plate_pattern, '', unfiltered_string.upper())
        
        if char_idx < len(self.character_set):
            raw_text.append(self.character_set[char_idx])
            confidences.append(confidence)
            
        # Enforce Colombian plate domain math (A-Z, 0-9)
        # filtered_string = re.sub(r'[^A-Z0-9]', '', unfiltered_string.upper())
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return filtered_string, avg_confidence


    def extract_text(self, processed_crop: np.ndarray) -> Tuple[str, float]:
        """
        End-to-end execution of the OCR model.
        
        Args:
            processed_crop (np.ndarray): The cropped license plate array.
            
        Returns:
            Tuple[str, float]: The license plate text and confidence.
        """
        tensor = self._preprocess_for_crnn(processed_crop)

        outputs = self.session.run(None, {self.input_name: tensor})
        raw_probabilities = outputs[0]
        
        raw_probabilities = outputs[0]

        if raw_probabilities.ndim == 4:
            # Remove extra channel dimension if present
            raw_probabilities = raw_probabilities[:,0,:,:]

        elif raw_probabilities.ndim == 3:
            # No channel dimension present
            pass

        else:
            raise ValueError(f"Unexpected OCR output shape: {raw_probabilities.shape}")


        # Remove extra channel dimension if present
         
        return self._ctc_decode(raw_probabilities)
