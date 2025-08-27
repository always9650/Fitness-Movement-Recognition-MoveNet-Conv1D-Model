import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, Flatten, Dense, Dropout, Lambda, Embedding
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import json
from datetime import datetime
import os
from sklearn.model_selection import train_test_split

# GPU 設置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

def triplet_center_loss(y_true, y_pred, n_classes=22, alpha=0.4):
    classes = tf.range(0, n_classes, dtype=tf.float32)
    y_pred_r = tf.reshape(y_pred, (tf.shape(y_pred)[0], 1))
    y_pred_r = tf.keras.backend.repeat(y_pred_r, n_classes)

    y_true_r = tf.reshape(y_true, (tf.shape(y_true)[0], 1))
    y_true_r = tf.keras.backend.repeat(y_true_r, n_classes)
    y_true_r = tf.cast(y_true_r, tf.float32)
    mask = tf.equal(y_true_r[:, :, 0], classes)

    masked = y_pred_r[:, :, 0] * tf.cast(mask, tf.float32)
    masked = tf.where(tf.equal(masked, 0.0), np.inf*tf.ones_like(masked), masked)
    minimums = tf.math.reduce_min(masked, axis=1)
    
    loss = K.max(y_pred - minimums + alpha, 0)
    return loss

def build_conv1d_model(learning_rate=0.001):
    # Input layer
    x_input = Input(shape=(34, 1))
    
    # CNN layers
    net = Conv1D(64, 3, padding='same', activation=tf.nn.relu6)(x_input)
    net = MaxPool1D(2, 2)(net)
    net = Conv1D(128, 3, padding='same', activation=tf.nn.relu6)(net)
    net = Conv1D(512, 3, padding='same', activation=tf.nn.relu6)(net)
    net = Conv1D(1024, 3, padding='same', activation=tf.nn.relu6)(net)
    net = Flatten()(net)
    net = Dense(256)(net)
    net = Dropout(0.5)(net)
    net = Dense(256)(net)
    net = Dense(128)(net)
    pre_logits = Dense(128, activation=tf.nn.relu6)(net)
    
    # Output layers
    softmax = Dense(22, activation='softmax', name='dense_4')(pre_logits)
    
    # Triplet center loss
    target_input = Input((1,), name='target_input')
    center = Embedding(22, 128)(target_input)
    l2_loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss')(
        [pre_logits, center])
    
    # Final model
    model = Model(inputs=[x_input, target_input], outputs=[softmax, l2_loss])
    
    # Compile with custom loss weights
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            'dense_4': 'categorical_crossentropy',
            'l2_loss': triplet_center_loss
        },
        loss_weights=[1.0, 0.5]
    )
    
    return model

def load_and_preprocess_data(json_path):
    with open(json_path, 'r+') as f:
        data = json.load(f)  

    frame = []
    X = []
    Y = []
    label_index = []
    for key in data.keys():
        bones_temp_list = []
        bbox = data[key]['bbox'].replace('[','').replace(']','').split(',')
        std_y, std_x, max_y, max_x = bbox[:4]
        std_x = float(std_x)
        std_y = float(std_y)
        max_x = float(max_x) - std_x
        max_y = float(max_y) - std_y

        for counter, value in enumerate(data[key]['bones'][1:-1].split(',')):
            value = float(value.replace('[', '').replace(']', ''))
            
            if counter % 3 == 0:
                value = (value - std_y) / max_y
            elif counter % 3 == 1:
                value = (value - std_x) / max_x 
            else:
                continue
            bones_temp_list.append(value)

        label_name = data[key]['label']
        if label_name not in label_index:
            label_index.append(label_name)
        label = label_index.index(label_name)
        
        frame.append(key)
        X.append(bones_temp_list)
        Y.append(label)

    X = np.array(X).reshape(-1, 34, 1)
    Y = np.array(Y)
    Y = np.uint8(Y)
    
    return X, Y

if __name__ == "__main__":
    # 訓練參數
    BATCH_SIZE = 128
    EPOCHS = 100000
    LEARNING_RATE = 1e-4
    HISTOGRAM_FREQ = 5000
    
    # 模型保存路徑
    best_model = "./model_triplet_center_loss"
    if not os.path.exists(best_model):
        os.makedirs(best_model)
    model_path = os.path.join(best_model, str(datetime.now().date()) + 'best_model.weights.h5')
    logdir = os.path.join(best_model, "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    # 加載數據
    X, Y = load_and_preprocess_data('./data/bones_label.json')
    
    # 數據分割
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
    le = LabelBinarizer()
    y_train_onehot = le.fit_transform(y_train)
    y_test_onehot = le.transform(y_test)
    
    # 構建模型
    model = build_conv1d_model(learning_rate=LEARNING_RATE)
    
    # 設置回調
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_path, save_weights_only=True, save_best_only=True, mode='min'),
        tf.keras.callbacks.TensorBoard(logdir, histogram_freq=HISTOGRAM_FREQ),
    ]
    
    # 開始訓練
    model.fit([x_train, y_train], 
              y=[y_train_onehot, y_train],
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_split=0.1,
              callbacks=callbacks)
