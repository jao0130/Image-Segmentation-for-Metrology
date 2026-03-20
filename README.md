# Animal Metrology Pipeline

使用 YOLOv8-seg 對 COCO 資料集進行動物分割，並以自訓練 YOLOv8-pose 模型量測雙眼距離與跨動物右眼距離。

## 技術棧

| 工具 | 用途 |
|------|------|
| YOLOv8-seg (ultralytics) | Instance segmentation |
| YOLOv8-pose (self-trained) | 眼睛 keypoint 偵測 |
| AP-10K | 眼睛 keypoint 訓練資料集 |
| OpenCV CLAHE | 深色動物預處理增強 |
| pandas | CSV 輸出 |
| PyTorch + CUDA | GPU 加速推論 |

## 專案結構

```
├── src/
│   ├── main.py                  # Pipeline 入口
│   ├── download_coco.py         # COCO 自動下載
│   ├── filter_images.py         # 多動物影像篩選
│   ├── segment.py               # YOLOv8-seg 分割
│   ├── measure.py               # 距離計算
│   ├── visualize.py             # 結果視覺化
│   └── eye_keypoint_model/
│       ├── prepare_dataset.py   # AP-10K → YOLOv8 格式轉換
│       ├── train.py             # 眼睛 keypoint 模型訓練
│       └── predict.py           # 眼睛偵測推論
├── architecture/
│   ├── architecture.png         # 系統架構圖
│   └── generate_architecture.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── .env.example
└── requirements.txt
```

## 本地運行

```bash
# 1. 安裝依賴（CUDA 12.x）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 2. 複製環境設定
cp .env.example .env

# 3. 執行 pipeline（自動下載 COCO ~1.2GB）
python src/main.py
```

結果輸出至 `output/images/` 和 `output/measurements.csv`。

## 眼睛 Keypoint 模型訓練

使用前需先訓練眼睛偵測模型：

```bash
# 1. 下載 AP-10K 資料集至 data/ap10k/
#    https://github.com/AlexTheBad/AP-10K

# 2. 轉換資料集格式
python src/eye_keypoint_model/prepare_dataset.py

# 3. 訓練模型
python src/eye_keypoint_model/train.py

# 4. 將訓練好的權重複製至 weights/
cp runs/keypoint/animal_eyes_v1/weights/best.pt weights/best.pt
```

## Docker 部署

```bash
cd docker
docker-compose up --build
```

## 系統架構

![架構圖](architecture/architecture.png)

## 量測公式

**單隻動物雙眼距離：**
```
d = sqrt((x_right - x_left)² + (y_right - y_left)²)
```

**兩隻動物右眼距離：**
```
d = sqrt((x_A_right - x_B_right)² + (y_A_right - y_B_right)²)
```

所有量測單位為 **像素（pixel）**。

## AI 模型說明

**分割模型：** `yolov8n-seg.pt`
- 預訓練於 COCO80，支援 10 種動物類別（bird、cat、dog、horse、sheep、cow、elephant、bear、zebra、giraffe）
- Box mAP@50: 52.0 / Mask mAP@50: 42.4（官方 COCO val2017）

**眼睛 Keypoint 模型：** YOLOv8-pose（自訓練）
- 訓練資料：AP-10K（17 keypoints，僅保留 left_eye、right_eye）
- 推論前套用 CLAHE 增強，改善深色毛色動物的眼睛偵測準確度
- 評估指標：Pose mAP@0.5（OKS）
