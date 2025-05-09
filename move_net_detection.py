import tensorflow_hub as hub
import tensorflow as tf
import os
import json
import cv2
import numpy as np

# MoveNet模型URL
MODEL_URL = "https://tfhub.dev/google/movenet/multipose/lightning/1"

def load_model():
    """載入MoveNet模型"""
    return hub.load(MODEL_URL)

def load_images_from_folders(base_path):
    """從資料夾結構讀取影像"""
    data = {}
    for label in os.listdir(base_path):
        label_path = os.path.join(base_path, label)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(label_path, filename)
                    data[filename] = {
                        'path': img_path,
                        'label': label
                    }
    return data

def preprocess_image(image_path):
    """預處理影像供模型使用"""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(image, dtype=tf.int32)
    return image

def run_inference(model, image):
    """執行姿勢估計"""
    outputs = model.signatures['serving_default'](image)
    return outputs['output_0'].numpy()[0]

def process_output(output, filename, label):
    """處理模型輸出結果"""
    keypoints = output[:, :51].reshape(17, 3)
    bbox = output[:, 51:56]
    
    return {
        'label': label,
        'bones': str(keypoints.tolist()),
        'bbox': str(bbox.tolist())
    }

def save_to_json(results, output_path):
    """儲存結果為JSON檔案"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def main():
    # 初始化模型
    model = load_model()
    
    # 讀取影像資料
    base_path = './data/post_label/'
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    image_data = load_images_from_folders(base_path)
    results = {}
    
    # 處理每張影像
    for filename, data in image_data.items():
        try:
            # 預處理影像
            image = preprocess_image(data['path'])
            
            # 執行推論
            output = run_inference(model, image)
            
            # 處理結果
            results[filename] = process_output(
                output, filename, data['label'])
                
        except Exception as e:
            print(f"處理 {filename} 時發生錯誤: {str(e)}")
    
    # 儲存結果
    save_to_json(results, './data/bones_label.json')
    print("處理完成，結果已儲存至 data/bones_label.json")

if __name__ == "__main__":
    main()
