import onnx

model = onnx.load("/app/app/ml/ml_models/plate_ocr.onnx")
onnx.checker.check_model(model)

print("Model OK")

