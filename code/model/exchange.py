from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO('/home/jetson/yolov8/home/wheeltec/yolov8_detect_mini_car/model/yolo11n_2.5k.pt')

# Export the model to TensorRT format
model.export(format="engine")  # creates 'yolo11n.engine'

