"""
Jupyter Notebook cell to download YOLOv8 models and export them to ONNX format.
Ensures compatibility with onnxruntime==1.17.1 by forcing opset=19.
"""
from ultralytics import YOLO
import os

# Define output directory mapped to the Docker volume
OUTPUT_DIR = "../ml_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def export_yolo_to_onnx(model_name: str, export_name: str, img_size: int = 640):
    """
    Loads a PyTorch YOLO model and exports it to ONNX with strict opset requirements.
    """
    print(f"Loading {model_name}.pt...")
    model = YOLO(f"{model_name}.pt")  # Downloads the .pt file if not present locally
    
    print(f"Exporting {export_name} to ONNX with opset=19...")
    # Export parameters:
    # format='onnx': Target framework
    # opset=19: Critical for avoiding Docker runtime mismatch
    # dynamic=False: Static batch sizing is faster for ONNX Runtime GPU
    # simplify=True: Runs onnx-simplifier to optimize the computational graph
    export_path = model.export(
        format="onnx",
        imgsz=img_size,
        opset=19,
        dynamic=False,
        simplify=True,
        name=export_name
    )
    
    # Move the exported file to the shared directory
    final_path = os.path.join(OUTPUT_DIR, f"{export_name}.onnx")
    os.rename(export_path, final_path)
    print(f"Successfully exported to: {final_path}\n")

if __name__ == "__main__":
    # 1. Base Vehicle Detector (Trained on COCO - detects cars, trucks, buses)
    # We use a 640x640 input resolution for the full frame
    export_yolo_to_onnx(
        model_name="yolov8n", 
        export_name="vehicle_detector", 
        img_size=640
    )
    
    # 2. License Plate Detector (Requires a fine-tuned model)
    # Note: For now, we assume you have 'yolov8n_plate.pt' locally from your Roboflow dataset
    # If not, you will need to train it first before running this block.
    # We use a smaller input resolution (e.g., 320x320) because it only processes the cropped vehicle.
    try:
        export_yolo_to_onnx(
            model_name="yolov8n_plate", 
            export_name="plate_detector", 
            img_size=320
        )
    except FileNotFoundError:
        print("WARNING: 'yolov8n_plate.pt' not found. You need to train or download the plate detection weights first.")
