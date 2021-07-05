import tensorflow as tf
import numpy as np
import pandas as pd
import os
import models
from prepareDataset import downloadDataset, dataDictionary
from prepareDataset import getDataframe
from extract_feature import getDataset_mfcc, getDataset_mel, getDataset_mfcc_delta
from models import CNN_model, inceptionModule, Inception, attRNN, DSCNN
from extract_feature import plot_confusion_matrix
from python_speech_features import mfcc, logfbank, delta
from IPython.display import Audio
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from analysisFunctions import plot_confusion_matrix
from constants import inv_categories


#for cuda particles
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
#-----------------------------------------------------------------
"""Get data"""

# Download data
downloadDataset(data_path='dataset/')

# Get dict with files and labels
dataDict = dataDictionary(data_path='dataset/', nWords=36)

trainDF = getDataframe(dataDict['train'])
valDF = getDataframe(dataDict['val'])
testDF = getDataframe(dataDict['test'], include_unknown=True) #background included

print("Train files: {}".format(trainDF.shape[0]))
print("Validation files: {}".format(valDF.shape[0]))
print("Test files: {}".format(testDF.shape[0]))

#-----------------------------------------------------------------
"""
Data Generation- Extracting Features and Data Split
Output Shape for Mel : (99, 40)
Output Shape for MFCC : (99, 13)
Output Shape for MFCC-Delta : (99, 13)
"""

BATCH_SIZE = 32
NUM_EXAMPLES = 80000

train_data, train_steps = getDataset_mfcc(
    df=trainDF[:NUM_EXAMPLES],
    batch_size=BATCH_SIZE,
    cache_file='train_cache',
    shuffle=True,
    nfilt=40,
    scale=True
) 

val_data, val_steps = getDataset_mfcc(
    df=valDF,
    batch_size=BATCH_SIZE,
    cache_file='val_cache',
    nfilt=40,
    shuffle=False,
    scale=True
)

test_data, test_steps = getDataset_mfcc(
    df=testDF,
    batch_size=BATCH_SIZE,
    cache_file='test_cache',
    shuffle=False,
    nfilt=40,
    scale=True
)
#-----------------------------------------------------------------
"""# Show Features Plot"""

f = dataDict['train']['files'][5]
sampling_rate, wave = wavfile.read(f)

if len(wave) < 16000:
    silence_part = np.random.randint(-30, -30, 16000-len(wave))
    wave = np.append(np.asarray(wave), silence_part)

mfcc_feat = mfcc(wave, samplerate=16000, winlen=0.025, winstep=0.01, highfreq=16000/2)
mfcc_delta = delta(mfcc_feat, N=2)
mfcc_delta2 = delta(mfcc_delta, N=2)

# print(mfcc_delta2.shape)
# fig = plt.figure(figsize=(10,10))
# plt.imshow(mfcc_delta2.T)
# plt.show()

#-----------------------------------------------------------------
"""Model Selection"""

lstm_model = attRNN()
gru_model = attRNN(rnn = 'GRU')

lstm_model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["sparse_categorical_accuracy"])
lstm_model.summary()

gru_model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["sparse_categorical_accuracy"])
gru_model.summary()

# model.compile(optimizer=tf.keras.optimizers.Adam(),
#                   loss="sparse_categorical_crossentropy",
#                   metrics=["sparse_categorical_accuracy"])

EPOCHS = 50

# Stop if the validation accuracy doesn't improve for 5 epochs
earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# Reduce LR on Plateau
reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1)

# Save best models
modelCheckpoint = tf.keras.callbacks.ModelCheckpoint(
    "Att_RNN_MFCC_35.h5",
    monitor='val_sparse_categorical_accuracy',
    mode='max',
    save_best_only=True
)

history_lstm = lstm_model.fit(train_data.repeat(),
               steps_per_epoch=train_steps,
               validation_data=val_data.repeat(),
               validation_steps=val_steps,
               epochs=EPOCHS,
               callbacks=[earlyStopping, reduceLR])

history_gru = gru_model.fit(train_data.repeat(),
               steps_per_epoch=train_steps,
               validation_data=val_data.repeat(),
               validation_steps=val_steps,
               epochs=EPOCHS,
               callbacks=[earlyStopping, reduceLR])

#---------------------------------------------------------------------
""" Train-Validation Accuracy and Loss Plotting"""
sns.set()

loss = history.history['loss']
val_loss = history.history['val_loss']

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,8))

ax1.plot(loss, label='train')
ax1.plot(val_loss, label='validation')
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.set_title('Model loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(acc, label='train')
ax2.plot(val_acc, label='validation')
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.set_title('Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.tight_layout()
fig.show()

#---------------------------------------------------------------------
""" Prediction and Test Section"""

# # CNN Model
# cnn_test = model.predict(test_data, verbose=1)
# cnn_y_pred = np.argmax(cnn_test, axis=1)
# y_true = testDF['labels'].tolist()
# print('2D CNN Accuracy: {:.4f}'.format(accuracy_score(cnn_y_pred, y_true)))

# # Inception Model
# inception_test = model.predict(test_data, verbose=1)
# inception_y_pred = np.argmax(inception_test, axis=1)
# y_true = testDF['labels'].tolist()
# print('Inception-based CNN Accuracy: {:.4f}'.format(accuracy_score(inception_y_pred, y_true)))

# # Attention RNN with LSTM
lstm_test = lstm_model.predict(test_data, verbose=1)
lstm_y_pred = np.argmax(lstm_test, axis=1)
y_true = testDF['labels'].tolist()
print('LSTM Accuracy: {:.4f}'.format(accuracy_score(lstm_y_pred, y_true)))

# Attention RNN with GRU
gru_test = gru_model.predict(test_data, verbose=1)
gru_y_pred = np.argmax(gru_test, axis=1)
y_true = testDF['labels'].tolist()
print('GRU Accuracy: {:.4f}'.format(accuracy_score(gru_y_pred, y_true)))

# # DSCNN Model
# dscnn_test = model.predict(test_data, verbose=1)
# dscnn_y_pred = np.argmax(dscnn_test, axis=1)
# y_true = testDF['labels'].tolist()
# print('DSCNN Accuracy: {:.4f}'.format(accuracy_score(dscnn_y_pred, y_true)))

#---------------------------------------------------------------------
"""Plotting Confusion Matrices"""

# # CM for CNN
# cnn_cm = confusion_matrix(cnn_y_pred, y_true)
# plot_confusion_matrix(cnn_cm, target_names=list(inv_categories.values())[:-1], normalize=True)

# # CM for Inception-based CNN
# inception_cm = confusion_matrix(inception_y_pred, y_true)
# plot_confusion_matrix(inception_cm, target_names=list(inv_categories.values())[:-1], normalize=True)

# CM for Attention RNN with LSTM
lstm_cm = confusion_matrix(lstm_y_pred, y_true)
plot_confusion_matrix(lstm_cm, target_names=list(inv_categories.values())[:-1], normalize=True)

# CM for Attention RNN with GRU
gru_cm = confusion_matrix(gru_y_pred, y_true)
plot_confusion_matrix(gru_cm, target_names=list(inv_categories.values())[:-1], normalize=True)

# # CM for DSCNN
# dscnn_cm = confusion_matrix(dscnn_y_pred, y_true)
# plot_confusion_matrix(dscnn_cm, target_names=list(inv_categories.values())[:-1], normalize=True)





