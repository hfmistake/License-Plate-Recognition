from ultralytics import YOLO
from predictions import Prediction


if __name__ == '__main__':
    vehicle_indexes = [2, 3, 5, 7]
    plate_index = [0]
    plate = {
        "video_source": "videos/5.mp4",
        "line": [100, 300, 1800, 300],
        "object_indices": plate_index,
        "directory": "results/results22"
    }
    normal = {
        "video_source": "videos/5.mp4",
        "line": [100, 300, 1800, 300],
        "object_indices": vehicle_indexes,
        "directory": "results/results22"
    }
    pre_trained = YOLO('models/yolov8n.pt')
    plate_model = YOLO('models/treino5.pt')
    normal_prediction = Prediction(pre_trained, normal)
    plate_prediction = Prediction(plate_model, plate)
    normal_prediction.visualize()