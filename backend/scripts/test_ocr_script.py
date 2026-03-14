# ==============================================================================
# Local Execution Test Block
# Run this directly in the terminal: python app/ml/pipelines/ocr_processor.py
# ==============================================================================
from app.ml.pipelines.ocr_inference import PlateOCRProcessor
import cv2
import os


if __name__ == "__main__":
    import sys
    
    # Temporarily add the backend directory to the Python path to resolve imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    sys.path.append(backend_dir)
    
    try:
        from app.ml.preprocessing import preprocess_for_ocr
        
        # Target the uploaded test image
        test_image_path = os.path.join(backend_dir, "test_result_01.jpg")
        model_path = os.path.join(backend_dir, "app", "ml", "ml_models", "plate_ocr.onnx")
        
        img = cv2.imread(test_image_path)
        
        if img is None:
            print(f"[ERROR] Could not load image at {test_image_path}")
        else:
            print("Applying CLAHE and binarization preprocessing...")
            processed_img = preprocess_for_ocr(img)
            
            print(f"Loading ONNX Model from {model_path}...")
            # We use CPU for a quick local test to avoid CUDA initialization overhead
            processor = PlateOCRProcessor(model_path, use_gpu=False)
            
            print("Running inference...")
            text, conf = processor.extract_text(processed_img)
            
            print("-" * 40)
            print(f"FINAL PLATE RESULT : {text}")
            print(f"CONFIDENCE SCORE   : {conf:.4f}")
            print("-" * 40)
            
    except Exception as e:
        print(f"[FATAL ERROR] Pipeline test failed: {e}")