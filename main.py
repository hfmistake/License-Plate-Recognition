from ultralytics import YOLO
from predictions import Prediction

if __name__ == '__main__':
    vehicle_indexes = [2, 3, 5, 7]
    entry_line = [100, 220, 1800, 220]
    exit_line = [0, 370, 1800, 370]
    data = {
        "video_source": "videos/9.mp4",
        "line": exit_line,
        "object_indices": vehicle_indexes,
        "directory": "results/results22",
        "entry": False,
        "pre_processing": True,
        "send_post": False,
        "preview": True
    }
    pre_trained = YOLO('models/weights/yolov8s.pt')
    plate_model = YOLO('models/custom_plate/treino5.pt')
    plate_prediction = Prediction(plate_model, pre_trained, data)
    plate_prediction.predict()
