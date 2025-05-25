import torch
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device

def run_detection(source):
    device = select_device('')
    model = DetectMultiBackend("best.pt", device=device, dnn=False)
    stride, names = model.stride, model.names

    dataset = LoadImages(source, img_size=640, stride=stride, auto=True)
    results = []
    result_img_path = None

    for path, img, im0s, vid_cap, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img[None]

        pred = model(img)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

        im0 = im0s.copy()
        data = []
        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], im0.shape).round()
            for *xyxy, conf, cls in pred:
                label = names[int(cls)]
                x1, y1, x2, y2 = map(int, xyxy)
                data.append({
                    "label": label,
                    "confidence": float(conf),
                    "xmin": x1,
                    "ymin": y1,
                    "xmax": x2,
                    "ymax": y2,
                })
                cv2.rectangle(im0, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(im0, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        out_path = Path("runs/detect/streamlit.jpg")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), im0)

        df = pd.DataFrame(data)
        df_path = out_path.parent / "hasil_deteksi.csv"
        df.to_csv(df_path, index=False)
        result_img_path = str(out_path)

    return result_img_path, str(df_path), df
