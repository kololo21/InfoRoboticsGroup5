"""
YOLOv8（ultralytics）で物体検出を行うモジュール

ultralytics は Python 3.8 以上が必要なので、このモジュールは /usr/bin/python3 と
/workspace/.yolo_env のライブラリで動かす（launch/yolo.launch と scripts/run_with_yolo_env.sh）。
"""
from wrs_detector.detector_utils import get_bbox_dict


class YoloDetector:
    """62401738 いしい
    YOLOv8 で物体検出を行うクラス。predict() の入出力は FasterRcnnDetector と同じ。
    """
    def __init__(self, weights_path, image_size=640, device=0):
        """62401738 いしい
        学習済みの重み（.pt）を読み込む。クラス名は重みに入っている名前（wod-c のラベル名）を使う。
        """
        from ultralytics import YOLO  # pylint: disable=import-outside-toplevel
        self.model = YOLO(weights_path)
        self.image_size = image_size
        self.device = device
        self.names = self.model.names

    def predict(self, img_np, threshold=0.5):
        """62401738 いしい
        YOLOv8 で物体検出を実行する

        Note
        --------
        * 画像の入力形式は uint8/BGR とする（ultralytics の numpy 入力と同じ）。
        """
        result = self.model.predict(img_np, imgsz=self.image_size, conf=threshold,
                                    device=self.device, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy().astype(int)

        result_array = []
        for box, label_id, score in zip(boxes, labels, scores):
            result_array.append(get_bbox_dict(self.names[label_id], float(score), box))
        return result_array
