
import os
import cv2
import numpy as np
import onnxruntime as ort
from typing import Tuple

class PlateOCRProcessor:
    
    def __init__(self, ocr_model_path: str, use_gpu: bool = True):
        """
        Initializes the ONNX Runtime session for the OCR model.
        
        Args:
            ocr_model_path (str): Path to the ONNX recognition model.
            use_gpu (bool): Hardware acceleration flag.
        """
        self.providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        
        if not os.path.exists(ocr_model_path):
            raise FileNotFoundError("OCR ONNX model not found.")

        self.session = ort.InferenceSession(ocr_model_path, providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name
        
        # Restricted dictionary for Colombian plates (A-Z, 0-9)
        # Index 0 is reserved for the CTC 'blank' token
        self.character_set = [
            'blank', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
        ]

    def _preprocess_for_crnn(self, processed_crop: np.ndarray) -> np.ndarray:
        """
        Resizes the preprocessed plate image to the fixed height required by the CRNN 
        (usually 32px or 48px) while maintaining the aspect ratio for the width.
        
        Args:
            processed_crop (np.ndarray): Image processed by CLAHE and thresholding.
            
        Returns:
            np.ndarray: NCHW tensor ready for inference.
        """
        target_height = 48
        h, w = processed_crop.shape[:2]
        
        # Calculate dynamic width based on the fixed height
        ratio = w / float(h)
        target_width = int(target_height * ratio)
        
        resized = cv2.resize(processed_crop, (target_width, target_height))
        
        # If the image was grayscale (2D), add the channel dimension back
        if len(resized.shape) == 2:
            resized = np.expand_dims(resized, axis=-1)
            
        # Normalize and transpose to NCHW
        tensor = resized.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        
        return tensor

    def _ctc_decode(self, predictions: np.ndarray) -> Tuple[str, float]:
        """
        Decodes the raw Softmax probabilities using CTC Greedy Decoding.
        Collapses repeated characters and ignores the 'blank' token.
        
        Args:
            predictions (np.ndarray): The raw output tensor from the CRNN.
            
        Returns:
            Tuple[str, float]: The decoded text and its average confidence score.
        """
        # Shape is usually (batch_size, sequence_length, num_classes)
        # We take the first batch item
        sequence = predictions[0] 
        
        text = []
        confidences = []
        previous_idx = 0
        
        # Iterate over the sequence time steps
        for step in sequence:
            char_idx = int(np.argmax(step))
            confidence = float(np.max(step))
            
            # CTC logic: ignore blanks (idx 0) and consecutive duplicates
            if char_idx != 0 and char_idx != previous_idx:
                text.append(self.character_set[char_idx])
                confidences.append(confidence)
                
            previous_idx = char_idx
            
        final_text = "".join(text)
        
        # Avoid division by zero if nothing was detected
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return final_text, avg_confidence

    def extract_text(self, processed_crop: np.ndarray) -> Tuple[str, float]:
        """
        End-to-end execution of the OCR model.
        
        Args:
            processed_crop (np.ndarray): The cropped license plate array.
            
        Returns:
            Tuple[str, float]: The license plate text and confidence.
        """
        tensor = self._preprocess_for_crnn(processed_crop)
        
        # Run the ONNX computational graph
        outputs = self.session.run(None, {self.input_name: tensor})
        raw_probabilities = outputs[0]
        
        return self._ctc_decode(raw_probabilities)