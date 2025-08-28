# 健身動作識別系統

## 專案簡介
本系統使用深度學習技術識別健身動作，包含三種模型架構：
1. MoveNet + Conv1D 模型
2. YOLOv4 模型  
3. VGG16 模型

資料集來自網路公開健身教學影片，包含23種健身動作，如原地慢跑、開合跳、深蹲等。

## 模型架構
### 1. MoveNet + Conv1D
- MoveNet模型輸入尺寸：256x160x3
- 輸出17個關節點座標
- Conv1D模型架構參考Martinez et al.(2020)
- 移除LSTM層，增加Triplet-Center損失函數

### 2. YOLOv4
- 輸入尺寸：416x416x3
- 批次大小：512
- 最大批次：132000
- 學習率：0.001

### 3. VGG16
- 輸入尺寸：224x398x3 
- 修改全連接層(FC1/FC2:1024神經元)
- 輸出層：22個類別
- 學習率：0.001

## 安裝指南
```bash
pip install -r requirements.txt
```

## 使用說明

### 完整工作流程
1. 下載影片
```bash
yt-dlp -f "bestvideo[ext=mp4]" -o ./data/video.mp4 https://www.youtube.com/watch?v=20uf1EcGqjY&ab_channel=EugeneWong
```

2. 影片轉換為圖片
```python
python video_to_images.py
```

3. 手動分類圖片
- 將 ./data/All_image/ 中的圖片手動分類至 ./data/post_label/<label>/ 目錄
- label 為動作類別名稱 

4. 執行 MoveNet 姿勢檢測
```python
python move_net_detection.py
```

5. 訓練模型
- Conv1D 模型:
```python
python conv1d_model.py
```

- VGG16 模型:
```python
python vgg16_model.py
```

- YOLOv4 模型:
需先安裝 YOLOv4 環境後執行訓練
```
```

Extra: TSNE
```python
python model_pred.py
```

## 資料集
- 訓練樣本：659張
- 驗證樣本：73張  
- 測試樣本：184張
- 總計：916張影像
- 23種健身動作類別
