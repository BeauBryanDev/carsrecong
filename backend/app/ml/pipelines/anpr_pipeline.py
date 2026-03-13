import time
import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Any, Optional
from pathlib import Path
from fastapi import UploadFile
from PIL import Image
import io

from app.core.config import settings 
from app.ml.preprocessing import preprocess_plate 

class ANPRPipeline:
    def __init__(
        self,
        vehicle_model_path: str = "/ml_models/vehicle_yolov10n.onnx",
        plate_model_path: str = "/ml_models/plate_yolov10n.onnx",
        ocr_model_path: str = "/ml_models/crnn.onnx",
        providers: List[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"],
    ):
        self.session_options = ort.SessionOptions()
        self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        available_providers = ort.get_available_providers()
        self.providers = [p for p in providers if p in available_providers]

        print(f"ONNX Runtime providers disponibles: {self.providers}")

        self.vehicle_session = ort.InferenceSession(vehicle_model_path, providers=self.providers, sess_options=self.session_options)
        self.plate_session = ort.InferenceSession(plate_model_path, providers=self.providers, sess_options=self.session_options)
        self.ocr_session = ort.InferenceSession(ocr_model_path, providers=self.providers, sess_options=self.session_options)
        self.vehicle_input_name = self.vehicle_session.get_inputs()[0].name
        self.plate_input_name = self.plate_session.get_inputs()[0].name
        self.ocr_input_name = self.ocr_session.get_inputs()[0].name

    def _preprocess_yolo(self, img: np.ndarray, target_size: int = 640) -> tuple:
        
        h, w = img.shape[:2]
        scale = min(target_size / w, target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        canvas = canvas.astype(np.float32) / 255.0
        canvas = np.transpose(canvas, (2, 0, 1))  # HWC -> CHW
        canvas = np.expand_dims(canvas, 0)  # add batch

        return canvas, (scale, (w, h))

    def _postprocess_yolo(
        self,
        outputs: np.ndarray,
        orig_shape: tuple,
        conf_thres: float = 0.4,
        iou_thres: float = 0.45,
        classes: Optional[List[int]] = None,
    ) -> List[Dict]:
        
        detections = []
        
        return detections

    async def process_image(self, file: UploadFile) -> Dict[str, Any]:
        start_time = time.time()

        contents = await file.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Imagen inválida")

        vehicle_input, scale_info = self._preprocess_yolo(img)
        vehicle_outputs = self.vehicle_session.run(None, {self.vehicle_input_name: vehicle_input})
        vehicles = self._postprocess_yolo(vehicle_outputs[0], img.shape[:2])

        results = []
        for veh in vehicles:
            if veh['class'] not in [0, 1, 2]:  # 0=CAR, 1=BUS, 2=TRUCK (ajusta según tu data.yaml)
                continue

            # car crop 
            x1, y1, x2, y2 = map(int, veh['bbox'])
            crop_veh = img[y1:y2, x1:x2]

            plate_input, _ = self._preprocess_yolo(crop_veh)
            plate_outputs = self.plate_session.run(None, {self.plate_input_name: plate_input})
            plates = self._postprocess_yolo(plate_outputs[0], crop_veh.shape[:2], conf_thres=0.5)

            for plate in plates:
                px1, py1, px2, py2 = map(int, plate['bbox'])
                crop_plate = crop_veh[py1:py2, px1:px2]

                # plate preprocessing
                processed_plate = preprocess_plate(crop_plate) 

                # ocr
                ocr_input = self._preprocess_ocr(processed_plate) 
                ocr_output = self.ocr_session.run(None, {self.ocr_input_name: ocr_input})
                plate_text, plate_conf = self._postprocess_ocr(ocr_output)

                # save to s3 (placeholder)
                s3_original = "uploads/original.jpg"  # Implement boto3 upload
                s3_plate = "crops/plate.jpg"

                results.append({
                    "vehicle_type": veh['class_name'], 
                    "vehicle_conf": veh['conf'],
                    "plate_text": plate_text,
                    "plate_conf": plate_conf,
                    "is_allowed": False, 
                    "original_image_s3_key": s3_original,
                    "cropped_plate_s3_key": s3_plate,
                    "processing_time_ms": (time.time() - start_time) * 1000,
                })

        return {"detections": results, "total_time_ms": (time.time() - start_time) * 1000}

    async def process_video(self, file: UploadFile, frame_skip: int = 5) -> Dict[str, Any]:
        """Procesamiento de video .mp4 (frame por frame con skip para velocidad)."""
        # Similar a process_image pero con cv2.VideoCapture
        # Placeholder completo - lo expandimos cuando estés listo
        pass  # Implementar con cap = cv2.VideoCapture(io.BytesIO(contents))

    def _preprocess_ocr(self, img: np.ndarray) -> np.ndarray:
        # Gray, resize a altura 32, normalize, CHW, batch
        pass  # Implementa según tu CRNN export

    def _postprocess_ocr(self, output: np.ndarray) -> tuple[str, float]:
        # CTC decode o greedy decode + softmax conf
        pass  # Placeholder

# Instancia global (carga una vez al startup)
anpr_pipeline = ANPRPipeline()