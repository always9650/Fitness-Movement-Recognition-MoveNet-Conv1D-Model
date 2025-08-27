from multiprocessing.dummy import active_children
from re import T
import tensorflow as tf 
from tensorflow import keras
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Lambda, Embedding
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelBinarizer
import json 
import numpy as np 
import matplotlib.pyplot as plt
from datetime import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.patheffects as PathEffects
import seaborn as sns


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


path = './data/bones_label.json'
with open(path, 'r+') as f:
  data = json.load(f)  


def scatter(outdir, x, labels, subtitle=None):
    # We choose a color palette with seaborn.
    max_label = np.max(labels) + 1
    palette = np.array(sns.color_palette("hls", max_label))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                    c=palette[labels.astype(np.int)])
    # plt.xlim(-25, 25)
    # plt.ylim(-25, 25)
    # ax.axis('off')
    # ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(max_label):
        # Position of each label.
        xtext, ytext = np.median(x[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    if subtitle != None:
        plt.suptitle(subtitle)

    plt.savefig(outdir + "/" + subtitle)


def tsne_plot(outdir, name, x_train, x_test, y_train, y_test):
    tsne = TSNE()
    train_tsne_embeds = tsne.fit_transform(x_train[:512])
    scatter(outdir, train_tsne_embeds, y_train[:512], "Samples from Train Data, {}".format(name))

    eval_tsne_embeds = tsne.fit_transform(x_test[:512])
    scatter(outdir, eval_tsne_embeds, y_test[:512], "Samples from Test Data, {}".format(name))

def cnn(input):
    net = keras.layers.Conv1D(64, 3, padding='same', activation=tf.nn.relu6)(input)
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
    # net = keras.layers.BatchNormalization()(net)
    # pre_logit = keras.layers.ReLU()(net)
    softmax =  keras.layers.Dense(22, activation=tf.nn.softmax)(pre_logit)
    
    return softmax, pre_logit


def triplet_center_loss(y_true, y_pred, n_classes= 22, alpha=0.4):
    """
    Implementation of the triplet loss function
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

    total_lenght = y_pred.shape.as_list()[-1]
    #     print('total_lenght=',  total_lenght)
    #     total_lenght =12

    # repeat y_true for n_classes and == np.arange(n_classes)
    # repeat also y_pred and apply mask
    # obtain min for each column min vector for each class
    

    
    # tf.print(tf.reduce_max(y_pred))

    classes = tf.range(0, n_classes,dtype=tf.float32)
    y_pred_r = tf.reshape(y_pred, (tf.shape(y_pred)[0], 1))
    y_pred_r = tf.keras.backend.repeat(y_pred_r, n_classes)

    y_true_r = tf.reshape(y_true, (tf.shape(y_true)[0], 1))
    y_true_r = tf.keras.backend.repeat(y_true_r, n_classes)
    y_true_r = tf.cast(y_true_r, tf.float32)
    mask = tf.equal(y_true_r[:, :, 0], classes)

    #mask2 = tf.ones((tf.shape(y_true_r)[0], tf.shape(y_true_r)[1]))  # todo inf

    # use tf.where(tf.equal(masked, 0.0), np.inf*tf.ones_like(masked), masked)

    masked = y_pred_r[:, :, 0] * tf.cast(mask, tf.float32) #+ (mask2 * tf.cast(tf.logical_not(mask), tf.float32))*tf.constant(float(2**10))
    
    masked = tf.where(tf.equal(masked, 0.0), np.inf*tf.ones_like(masked), masked)
    minimums = tf.math.reduce_min(masked, axis=1)
    
    # tf.print(y_pred)
    # print('_'*30)
    # tf.print(minimums)
    # print('_'*20)
    # tf.print(alpha)
    # print('_'*10)
    loss = K.max(y_pred - minimums + alpha ,0)

    # obtain a mask for each pred
    # tf.print(loss)

    return loss


frame = []
X = []
Y = []
label_index = []
for key in data.keys():
  bones_temp_list = []
  bbox = data[key]['bbox'].replace('[','').replace(']','').split(',')
  # std_y, std_x = bbox[0:2]
  std_y, std_x, max_y, max_x = bbox[:4]
  std_x = float(std_x)
  std_y = float(std_y)
  max_x = float(max_x) - std_x
  max_y = float(max_y) - std_y
  # for in data[Key]['bbox']:
  

  for counter, value in enumerate(data[key]['bones'][1:-1].split(',')):
    value = float(value.replace('[', '').replace(']', ''))
    
    if counter % 3 == 0:
      value = (value - std_y) / max_y
    elif counter % 3 == 1:
      value = (value - std_x) / max_x 
    else:
      continue
    bones_temp_list.append(value)
    # data[key]['bones'] = np.array(bones_temp_list)

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

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
le = LabelBinarizer()
y_train_onehot = le.fit_transform(y_train)
y_test_onehot = le.fit_transform(y_test)


loss_weights = [1, 0.5]
embedding_size = 128
LEARNING_RATE = 1e-4

best_model = "./classification_model"
name = '2025-08-27'
model_path = os.path.join(best_model, name + 'best_model.weights.h5')
plot_dir = os.path.join('./plot', name)
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

x_input = Input(shape=(34, 1))

softmax, pre_logits = cnn(x_input,)

target_input = Input((1,), name='target_input')

center = Embedding(22, embedding_size)(target_input)
l2_loss = Lambda(lambda x: K.sum(K.square(x[0] - x[1][:, 0]), 1, keepdims=True), name='l2_loss')(
    [pre_logits, center])

# model = tf.keras.models.Model(inputs=[x_input, target_input], outputs=[softmax, l2_loss])



# model = tf.keras.models.Model(inputs=[x_input, target_input], outputs=[softmax, l2_loss])
# model.compile(loss=["categorical_crossentropy", triplet_center_loss],
#                   optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE), metrics=["accuracy"],
#                   loss_weights=loss_weights)

model = Model(inputs=[x_input, target_input], outputs=[softmax, l2_loss, pre_logits])
model.load_weights(model_path)

# evaluation = model.evaluate(x=[x_train, y_train], y=[y_train_onehot, y_train])
# print(evaluation)

_, x_b, X_train_embed = model.predict([x_train[:512], y_train[:512]])
a, b, X_test_embed = model.predict([x_test, y_test])

label = set(np.arange(22))

train_stat = set(y_train[:512]) == label
test_stat = set(y_test) == label


if train_stat and test_stat:

  a = np.argmax(a, axis=1)

  for i in range(22):
    index = np.where(y_test == i)[0]
    
    acc = np.where(a[index] == y_test[index], 1, 0)
    print(f'{i}, acc: {sum(acc) / len(index)}, size: {len(index)}')

  tsne_plot(plot_dir, "triplet_center_loss", X_train_embed, X_test_embed, y_train, y_test)