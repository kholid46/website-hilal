
import torch
import pandas as pd
from pathlib import Path
from PIL import Image

def compress_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((640, 640))
    temp = image_path + "_resized.jpg"
    img.save(temp, "JPEG")
    return temp

def run_detection(source_path):
    model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt", force_reload=False)
    model.conf = 0.25

    img_path = compress_image(source_path)
    results = model(img_path)
    results.save()

    out_dir = Path("runs/detect")
    latest = sorted(out_dir.glob("exp*"), key=lambda x: x.stat().st_mtime)[-1]
    result_img = list(latest.glob("*.jpg"))[0]

    data = []
    for pred in results.pred:
        if pred is not None and len(pred):
            for *box, conf, cls in pred.tolist():
                data.append({
                    "label": model.names[int(cls)],
                    "confidence": round(conf, 3),
                    "xmin": round(box[0]),
                    "ymin": round(box[1]),
                    "xmax": round(box[2]),
                    "ymax": round(box[3]),
                })

    df = pd.DataFrame(data)
    csv_path = latest / "hasil_deteksi.csv"
    df.to_csv(csv_path, index=False)

    return str(result_img), str(csv_path), df
