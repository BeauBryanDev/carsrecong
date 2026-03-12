import onnxruntime as ort
import os

# Configure the session options for optimal performance on CPU
options = ort.SessionOptions()

# Configure the number of threads for optimal performance on CPU
# Reserve 1 or 2 threads for the system/Postgres and the rest for the model.
options.intra_op_num_threads = 4  
options.inter_op_num_threads = 4

# Enable all hardware optimizations
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Load the model
model_path = "/ml_models/yolov8s.onnx"

session = ort.InferenceSession(
    model_path, 
    sess_options=options, 
    providers=['CPUExecutionProvider']
)

print(f"Model loaded successfully. Threads assigned: {options.intra_op_num_threads}")
