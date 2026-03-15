import os
import sys
import cv2
import numpy as np

# 1. Resolve absolute paths dynamically to inject the backend root into Python's path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
sys.path.append(backend_dir)

# Now we can safely import from our app modules
from app.ml.pipelines.inference import VehicleDetectionPipeline
from app.ml.pipelines.ocr_processor import PlateOCRProcessor
from app.ml.preprocessing import extract_plate_crop, preprocess_for_ocr

def run_pipeline_test():
    """Executes the cascading ML models on a single RAW test image."""
    
    # We enforce the use of a raw, unedited street image for an end-to-end test.
    image_path = os.path.join(backend_dir, "./test_vehicle_01.jpeg")
    
    print(f"[INFO] Loading image from: {image_path}")
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"[FATAL ERROR] Image not found at {image_path}. Please provide a valid raw image.")
        return

    print("[INFO] Initializing ONNX Runtime Sessions (CPU Mode)...")
    # Path resolution is now handled internally by the classes
    vehicle_pipeline = VehicleDetectionPipeline(use_gpu=False)
    ocr_processor = PlateOCRProcessor(use_gpu=False)
    
    # ---------------------------------------------------------
    # STAGE 1: Vehicle Detection
    # ---------------------------------------------------------
    print("[INFO] Running Stage 1: Vehicle Detection...")
    vehicles = vehicle_pipeline.detect_vehicles(frame, conf_threshold=0.5)
    
    if not vehicles:
        print("[WARNING] No vehicles detected in the image.")
        return
    
    for idx, vehicle in enumerate(vehicles):
        print(f"\n--- Processing Vehicle {idx + 1} ({vehicle['class']}) ---")
        
        # ---------------------------------------------------------
        # STAGE 2: License Plate Detection (Crop & Infer)
        # ---------------------------------------------------------
        # Extract the vehicle crop with a 12% padding mapping
        vehicle_crop = extract_plate_crop(frame, vehicle['bbox'], padding=0.12)
        
        # Preprocess tensor for the Nano plate detector (static size 320x320)
        tensor, scale = vehicle_pipeline._preprocess_image(vehicle_crop, target_size=320)
        
        print("[INFO] Running Stage 2: License Plate Detection...")
        plate_outputs = vehicle_pipeline.session_plate.run(None, {vehicle_pipeline.input_name_plate: tensor})
        plate_preds = np.squeeze(plate_outputs[0]).transpose()
        
        best_plate_box = None
        highest_conf = 0.0
        
        # Parse standard YOLOv8 output for the single 'plate' class
        for row in plate_preds:
            conf = float(row[4])
            if conf > 0.45 and conf > highest_conf:
                highest_conf = conf
                xc, yc, w, h = row[0], row[1], row[2], row[3]
                
                # Transform coordinates back to the vehicle_crop spatial domain
                x_min = int((xc - w / 2) / scale)
                y_min = int((yc - h / 2) / scale)
                x_max = int((xc + w / 2) / scale)
                y_max = int((yc + h / 2) / scale)
                best_plate_box = [x_min, y_min, x_max, y_max]
                
        if not best_plate_box:
            print("[WARNING] No license plate found on this vehicle.")
            continue
            
        # Extract the tight plate crop from the vehicle crop (no padding here)
        plate_crop = extract_plate_crop(vehicle_crop, best_plate_box, padding=0.0)
        
        # ---------------------------------------------------------
        # STAGE 3: OCR and Decoding
        # ---------------------------------------------------------
        print("[INFO] Running Stage 3: CLAHE Preprocessing & OCR...")
        processed_plate = preprocess_for_ocr(plate_crop)
        
        text, conf = ocr_processor.extract_text(processed_plate)
        
        print("=" * 40)
        print(f"FINAL RESULT - VEHICLE {idx + 1}")
        print(f"PLATE TEXT      : {text}")
        print(f"OCR CONFIDENCE  : {conf:.4f}")
        print("=" * 40)



if __name__ == "__main__":
    
    
    run_pipeline_test()