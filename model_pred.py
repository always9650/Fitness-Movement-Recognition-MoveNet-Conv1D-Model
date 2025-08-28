import json
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda, Embedding
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE


def configure_gpu():
    """Configures GPU memory growth for TensorFlow."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)


configure_gpu()


def load_json_data(file_path):
    """Loads JSON data from the specified file path."""
    with open(file_path, 'r+') as f:
        data = json.load(f)
    return data


data = load_json_data('./data/bones_label.json')


def _create_scatter_figure(x, labels, palette):
    """Helper to create a scatter plot figure."""
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[labels.astype(int)])

    txts = []
    for i in range(np.max(labels) + 1):
        xtext, ytext = np.median(x[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    return f


def save_plot_to_file(fig, output_directory, filename):
    """Saves a matplotlib figure to a specified file."""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    fig.savefig(os.path.join(output_directory, filename))
    plt.close(fig) # Close the figure to free up memory


def generate_scatter_plot(output_directory, x, labels, subtitle=None):
    """Generates and saves a scatter plot."""
    max_label = np.max(labels) + 1
    palette = np.array(sns.color_palette("hls", max_label))
    
    fig = _create_scatter_figure(x, labels, palette)
    if subtitle:
        plt.suptitle(subtitle)
    
    filename = f"{subtitle}.png" if subtitle else "scatter_plot.png"
    save_plot_to_file(fig, output_directory, filename)


def generate_tsne_plots(output_directory, name, x_train, x_test, y_train, y_test):
    """Generates and saves t-SNE plots for train and test data."""
    tsne = TSNE(random_state=42) # Added random_state for reproducibility
    
    train_tsne_embeds = tsne.fit_transform(x_train[:512])
    generate_scatter_plot(output_directory, train_tsne_embeds, y_train[:512],
                          f"Samples from Train Data, {name}")

    eval_tsne_embeds = tsne.fit_transform(x_test[:512])
    generate_scatter_plot(output_directory, eval_tsne_embeds, y_test[:512],
                          f"Samples from Test Data, {name}")


def build_cnn_layers(input_tensor):
    """Builds the CNN layers for the model."""
    net = keras.layers.Conv1D(64, 3, padding='same', activation=tf.nn.relu6)(input_tensor)
    net = keras.layers.MaxPool1D(2, 2)(net)
    net = keras.layers.Conv1D(128, 3, padding='same', activation=tf.nn.relu6)(net)
    net = keras.layers.Conv1D(512, 3, padding='same', activation=tf.nn.relu6)(net)
    net = keras.layers.Conv1D(1024, 3, padding='same', activation=tf.nn.relu6)(net)
    net = keras.layers.Flatten()(net)
    net = keras.layers.Dense(256)(net)
    net = keras.layers.Dropout(0.5)(net)
    net = keras.layers.Dense(256)(net)
    net = keras.layers.Dense(128)(net)
    pre_logit = keras.layers.Dense(128, activation=tf.nn.relu6)(net)
    softmax = keras.layers.Dense(22, activation=tf.nn.softmax)(pre_logit)
    return softmax, pre_logit


def triplet_center_loss(y_true, y_pred, n_classes=22, alpha=0.4):
    """
    Implementation of the triplet loss function.

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
    print('y_pred.shape = ', y_pred)

    classes = tf.range(0, n_classes, dtype=tf.float32)
    y_pred_r = tf.reshape(y_pred, (tf.shape(y_pred)[0], 1))
    y_pred_r = tf.keras.backend.repeat(y_pred_r, n_classes)

    y_true_r = tf.reshape(y_true, (tf.shape(y_true)[0], 1))
    y_true_r = tf.keras.backend.repeat(y_true_r, n_classes)
    y_true_r = tf.cast(y_true_r, tf.float32)
    mask = tf.equal(y_true_r[:, :, 0], classes)

    masked = y_pred_r[:, :, 0] * tf.cast(mask, tf.float32)
    masked = tf.where(tf.equal(masked, 0.0), np.inf * tf.ones_like(masked), masked)
    minimums = tf.math.reduce_min(masked, axis=1)
    
    loss = K.max(y_pred - minimums + alpha, 0)
    return loss


def normalize_bbox_coordinates(bbox_str):
    """Parses and normalizes bounding box coordinates."""
    bbox = [float(val) for val in bbox_str.replace('[', '').replace(']', '').split(',')]
    std_y, std_x, max_y, max_x = bbox[:4]
    max_x -= std_x
    max_y -= std_y
    return std_x, std_y, max_x, max_y


def process_bones_data(bones_str, std_x, std_y, max_x, max_y):
    """Processes and normalizes bone coordinates."""
    bones_temp_list = []
    for counter, value_str in enumerate(bones_str[1:-1].split(',')):
        value = float(value_str.replace('[', '').replace(']', ''))
        if counter % 3 == 0:
            value = (value - std_y) / max_y
        elif counter % 3 == 1:
            value = (value - std_x) / max_x
        else:
            continue
        bones_temp_list.append(value)
    return bones_temp_list


def prepare_dataset(raw_data):
    """Prepares the dataset from raw JSON data."""
    all_bones_data = []
    all_labels = []
    label_index_map = {}
    
    for key in raw_data.keys():
        bbox_str = raw_data[key]['bbox']
        std_x, std_y, max_x, max_y = normalize_bbox_coordinates(bbox_str)
        
        bones_str = raw_data[key]['bones']
        bones_temp_list = process_bones_data(bones_str, std_x, std_y, max_x, max_y)
        
        label_name = raw_data[key]['label']
        if label_name not in label_index_map:
            label_index_map[label_name] = len(label_index_map)
        label = label_index_map[label_name]
        
        all_bones_data.append(bones_temp_list)
        all_labels.append(label)

    X = np.array(all_bones_data).reshape(-1, 34, 1)
    Y = np.array(all_labels, dtype=np.uint8)
    return X, Y, list(label_index_map.keys())


def split_and_binarize_data(X, Y, test_size=0.2, shuffle=True):
    """Splits data into train/test sets and binarizes labels."""
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, shuffle=shuffle, random_state=42) # Added random_state for reproducibility
    le = LabelBinarizer()
    y_train_onehot = le.fit_transform(y_train)
    y_test_onehot = le.transform(y_test) # Use transform for test set
    return x_train, x_test, y_train, y_test, y_train_onehot, y_test_onehot


def create_model(input_shape, num_classes, embedding_size, learning_rate):
    """Creates and compiles the Keras model."""
    x_input = Input(shape=input_shape)
    softmax_output, pre_logits_output = build_cnn_layers(x_input)

    target_input = Input((1,), name='target_input')
    center = Embedding(num_classes, embedding_size)(target_input)
    l2_loss_output = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss')(
        [pre_logits_output, center])

    model = Model(inputs=[x_input, target_input], outputs=[softmax_output, l2_loss_output, pre_logits_output])
    
    return model


def main():
    """Main function to execute the model prediction and evaluation pipeline."""
    # Data Loading and Preprocessing
    raw_data = load_json_data('./data/bones_label.json')
    X, Y, label_names = prepare_dataset(raw_data)
    x_train, x_test, y_train, y_test, y_train_onehot, y_test_onehot = split_and_binarize_data(X, Y)

    # Model Parameters
    num_classes = len(label_names)
    embedding_size = 128
    learning_rate = 1e-4
    input_shape = (X.shape[1], X.shape[2])

    best_model_dir = "./model_triplet_center_loss"
    current_date_name = '2025-08-27' # Consider using datetime.now().strftime('%Y-%m-%d') for dynamic naming
    model_weights_path = os.path.join(best_model_dir, current_date_name + 'best_model.weights.h5')
    plot_output_dir = os.path.join('./plot', current_date_name)
    
    if not os.path.exists(plot_output_dir):
        os.makedirs(plot_output_dir)

    # Model Creation and Loading Weights
    model = create_model(input_shape, num_classes, embedding_size, learning_rate)
    model.load_weights(model_weights_path)

    # Predictions
    _, _, x_train_embed = model.predict([x_train[:512], y_train[:512]])
    softmax_predictions, _, x_test_embed = model.predict([x_test, y_test])

    # Evaluation and Plotting
    unique_labels_train = set(y_train[:512])
    unique_labels_test = set(y_test)
    all_possible_labels = set(np.arange(num_classes))

    if unique_labels_train == all_possible_labels and unique_labels_test == all_possible_labels:
        predicted_classes = np.argmax(softmax_predictions, axis=1)

        for i in range(num_classes):
            indices = np.where(y_test == i)[0]
            if len(indices) > 0:
                accuracy = np.mean(predicted_classes[indices] == y_test[indices])
                print(f'Class {i} ({label_names[i]}), Accuracy: {accuracy:.4f}, Size: {len(indices)}')
            else:
                print(f'Class {i} ({label_names[i]}), No samples in test set.')

        generate_tsne_plots(plot_output_dir, "triplet_center_loss", x_train_embed, x_test_embed, y_train, y_test)
    else:
        print("Warning: Not all classes are present in train or test subsets for full evaluation/plotting.")
        print(f"Train labels present: {sorted(list(unique_labels_train))}")
        print(f"Test labels present: {sorted(list(unique_labels_test))}")


if __name__ == "__main__":
    main()
