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
    def __init__(self, weights_path, image_size=640, device=None):
        """62401738 いしい
        学習済みの重み（.pt）を読み込む。クラス名は重みに入っている名前（wod-c のラベル名）を使う。

        device を省略すると、GPU があれば GPU(0)、なければ CPU を使う（GPU のない PC で
        "Invalid CUDA device=0" で落ちないように）。環境変数 YOLO_DEVICE（cpu または 0）と
        YOLO_IMAGE_SIZE（例 416。小さいほど CPU で速いが精度は下がる）で上書きできる。
        """
        import os  # pylint: disable=import-outside-toplevel
        from ultralytics import YOLO  # pylint: disable=import-outside-toplevel
        self.model = YOLO(weights_path)
        self.image_size = int(os.environ.get("YOLO_IMAGE_SIZE", image_size))
        if device is None:
            device = os.environ.get("YOLO_DEVICE", "")
            if device.isdigit():
                device = int(device)
            elif not device:
                import torch  # pylint: disable=import-outside-toplevel
                device = 0 if torch.cuda.is_available() else "cpu"
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
