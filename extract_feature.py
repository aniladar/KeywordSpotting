import os

import numpy as np
import tensorflow as tf
from python_speech_features import logfbank, mfcc, delta
from scipy.io import wavfile

AUDIO_LENGTH = 16000
AUDIO_SR = 16000


def getDataset_mel(df, batch_size, cache_file=None, shuffle=True, parse_param=(0.025, 0.01, 40), scale=False):
    """
    :param df: dataframe
    :param batch_size: batch size for audio file
    :param cache_file: cache for tf slice
    :param shuffle: shuffle option for dataset for training
    :param parse_param: params for winlen, winstep and numcep
    :param scale: scaling
    :return: Return a tf.data.Dataset containg mel, labels
    """
    data = tf.data.Dataset.from_tensor_slices(
        (df['files'].tolist(), df['labels'].tolist())
    )

    data = data.map(
        lambda filename, label: tuple(
            tf.py_function(
                logMelFilterbank,
                inp=[filename, label, parse_param, scale],
                Tout=[tf.float32, tf.int32]
                )
        ),
        num_parallel_calls=os.cpu_count()
    )

    if cache_file:
        data = data.cache(cache_file)

    if shuffle:
        data = data.shuffle(buffer_size=df.shape[0])

    data = data.batch(batch_size).prefetch(buffer_size=1)

    steps = df.shape[0] // batch_size

    return data, steps


def getDataset_mfcc(df, batch_size, cache_file=None, shuffle=True, nfilt=40, scale=False):
    """
    :param df: dataframe
    :param batch_size: batch size for audio file
    :param cache_file: cache for tf slice
    :param shuffle: shuffle option for dataset for training
    :param nfilt: filter size for mfcc
    :param scale: scaling
    :return: Return a tf.data.Dataset containg MFCC, labels
    """
    data = tf.data.Dataset.from_tensor_slices(
        (df['files'].tolist(), df['labels'].tolist())
    )

    data = data.map(
        lambda filename, label: tuple(
            tf.py_function(
                mfcc_,
                inp=[filename, label, nfilt, scale],
                Tout=[tf.float32, tf.int32]
                )
        ),
        num_parallel_calls=os.cpu_count()
    )

    if cache_file:
        data = data.cache(cache_file)

    if shuffle:
        data = data.shuffle(buffer_size=df.shape[0])

    data = data.batch(batch_size).prefetch(buffer_size=1)

    steps = df.shape[0] // batch_size

    return data, steps

def getDataset_mfcc_delta(df, batch_size, cache_file=None, shuffle=True, nfilt=40, scale=False):
    """
    :param df: dataframe
    :param batch_size: batch size for audio file
    :param cache_file: cache for tf slice
    :param shuffle: shuffle option for dataset for training
    :param nfilt: filter size for mfcc delta
    :param scale: scaling
    :return: Return a tf.data.Dataset containg MFCC_delta, labels
    """
    data = tf.data.Dataset.from_tensor_slices(
        (df['files'].tolist(), df['labels'].tolist())
    )

    data = data.map(
        lambda filename, label: tuple(
            tf.py_function(
                mfcc_delta,
                inp=[filename, label, nfilt, scale],
                Tout=[tf.float32, tf.int32]
                )
        ),
        num_parallel_calls=os.cpu_count()
    )

    if cache_file:
        data = data.cache(cache_file)

    if shuffle:
        data = data.shuffle(buffer_size=df.shape[0])

    data = data.batch(batch_size).prefetch(buffer_size=1)

    steps = df.shape[0] // batch_size

    return data, steps


def _loadWavs(filename):
    """
    :param filename: takes raw audio file
    :return: Return a np array containing the wav.
    If len(wav) < AUDIO_LENGTH perform padding
    """
    _, wave = wavfile.read(filename)
    # pad
    if len(wave) < AUDIO_LENGTH:
        silence_part = np.random.normal(0, 5, AUDIO_LENGTH-len(wave))
        wave = np.append(np.asarray(wave), silence_part)

    return wave.astype(np.float32)


def _logMelFilterbank(wave, parse_param=(0.025, 0.01, 40)):
    """
    :param wave: audio signal
    :param parse_param: params for winlen, winstep and numcep
    :return: melfilterbank
    """
    fbank = logfbank(
        wave,
        samplerate=16000,
        winlen=float(parse_param[0]),
        winstep=float(parse_param[1]),
        highfreq=AUDIO_SR/2,
        nfilt=int(parse_param[2])
        )

    fbank = fbank.astype(np.float32)
    return fbank


def _mfcc(wave, nfilt=40):
    """
    Perform MFCC operation
    :param wave: audio signal
    :param nfilt: filter size
    :return: mfcc
    """
    mfcc_ = mfcc(
        wave,
        samplerate=16000,
        winlen=0.025,
        winstep=0.01,
        highfreq=AUDIO_SR/2,
        nfilt=nfilt
        )

    mfcc_ = mfcc_.astype(np.float32)
    return mfcc_

def _mfcc_delta(wave):
    """
    Perform MFCC derivative
    :param wave: audio signal
    :return: derivative of MFCC
    """
    mfcc_delta = delta(
      wave,
      N = 2
      )
    
    mfcc_delta = mfcc_delta.astype(np.float32)
    return mfcc_delta


def _normalize(data):
    """
    Normalize feature vectors
    :param data: audio
    :return: normalized features
    """
    mean = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    return (data - mean) / (sd+1e-08)


def _scale(data):
    """
    Scale input values in range [0,1]
    :param data: audio file
    :return: scaled input
    """
    min_value, max_value = np.min(data, axis=0), np.max(data, axis=0)
    scaled = (data - min_value) / (max_value - min_value + 1e-08)
    return scaled


def logMelFilterbank(filename, label, parse_param=(0.025, 0.01, 40), scale=False):
    """
    Function used to compute mel from file name.
    :param filename: filename for audio
    :param label: label for audio
    :param parse_param: winlen winstep and nfilt
    :param scale: scaling
    :return: Returns mel and label
    """
    wave = _loadWavs(filename.numpy())
    fbank = _logMelFilterbank(wave, parse_param)
    if scale:
        fbank = _normalize(fbank)
    return fbank, np.asarray(label).astype(np.int32)


def mfcc_(filename, label, nfilt=40, scale=False):
    """
    Function used to compute mfcc from file name.
    :param filename: filename for audio
    :param label: label for audio
    :param scale: scaling
    :return: Returns mfcc and label
    """
    wave = _loadWavs(filename.numpy())
    mfcc_ = _mfcc(wave, nfilt)
    if scale:
        mfcc_ = _normalize(mfcc_)
    return mfcc_, np.asarray(label).astype(np.int32)

def mfcc_delta(filename, label, nfilt=40, scale=False):
    """
    Function used to compute mfcc-delta from file name.
    :param filename: filename for audio
    :param label: label for audio
    :param scale: scaling
    :return: Returns mfcc-delta and label
    """
    wave = _loadWavs(filename.numpy())
    mfcc_ = _mfcc(wave, nfilt)
    mfcc_delta = _mfcc_delta(mfcc_)
    mfcc_delta2 = _mfcc_delta(mfcc_delta)
    
    if scale:
        mfcc_delta2 = _normalize(mfcc_delta2)
    return mfcc_delta2, np.asarray(label).astype(np.int32)

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):
    """
    given a sklearn confusion matrix (cm), make a nice plot
    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix
    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']
    title:        the text to display at the top of the matrix
    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues
    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph
    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    # plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=15)
        plt.yticks(tick_marks, target_names, fontsize=15)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 10 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.3f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass), fontsize=30)
    plt.show()
