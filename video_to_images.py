import cv2
import os

video_path = "./data/video.mp4"
output_dir = "./data/All_image"

# 自動建立輸出目錄
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 儲存幀為圖片，使用4位數編號
    output_path = f"{output_dir}/{frame_count}.jpg"
    cv2.imwrite(output_path, frame)
    frame_count += 1

cap.release()
print(f"轉換完成，共 {frame_count} 張圖片儲存於 {output_dir}")
