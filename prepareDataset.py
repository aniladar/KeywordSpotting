import os
import requests
import tarfile
import pandas as pd


def downloadDataset(data_path='dataset/'):
    """This function downloads the Google Speech Command Dataset v2
    If the dataset already exists in the folder, it gives an error.
    @param data_path: Takes data path to download the dataset.
    """

    data_path = os.path.abspath(data_path)+'/'
    datasets = ['train', 'test']
    urls = [
        'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
        'http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz'
    ]

    for dataset, url in zip(datasets, urls):
        dataset_directory = data_path + dataset
        # Check if we need to extract the dataset
        if not os.path.isdir(dataset_directory):
            os.makedirs(dataset_directory)
            file_name = data_path + dataset + '.tar.gz'
            # Check if the dataset has been downloaded
            if os.path.isfile(file_name):
                print('{} already exists. Skipping download.'.format(file_name))
            else:
                downloadFile(url=url, file_name=file_name)

            # extract downloaded file
            extractFile(file_name=file_name, directory=dataset_directory)
        else:
            print('Nothing to do.')


def dataDictionary(data_path='dataset/', nWords=36):

    global words, inv_words
    
    # Hold words and number of words   
    if nWords == 36:
     words = {
        '_background_noise_': 35,
        'bed': 16,
        'bird': 7,
        'cat': 12,
        'dog': 8,
        'down': 21,
        'eight': 5,
        'five': 20,
        'four': 3,
        'go': 27,
        'happy': 18,
        'house': 26,
        'left': 13,
        'marvin': 22,
        'nine': 1,
        'no': 9,
        'off': 2,
        'on': 10,
        'one': 6,
        'right': 4,
        'seven': 11,
        'sheila': 19,
        'six': 23,
        'stop': 0,
        'three': 14,
        'tree': 15,
        'two': 29,
        'up': 24,
        'wow': 25,
        'yes': 28,
        'zero': 17,
        'backward': 30,
        'follow': 31,
        'forward': 32,
        'learn': 33,
        'visual': 34}

     inv_words = {
        0: 'stop',
        1: 'nine',
        2: 'off',
        3: 'four',
        4: 'right',
        5: 'eight',
        6: 'one',
        7: 'bird',
        8: 'dog',
        9: 'no',
        10: 'on',
        11: 'seven',
        12: 'cat',
        13: 'left',
        14: 'three',
        15: 'tree',
        16: 'bed',
        17: 'zero',
        18: 'happy',
        19: 'sheila',
        20: 'five',
        21: 'down',
        22: 'marvin',
        23: 'six',
        24: 'up',
        25: 'wow',
        26: 'house',
        27: 'go',
        28: 'yes',
        29: 'two',
        30: 'backward',
        31: 'follow',
        32: 'forward',
        33: 'learn',
        34: 'visual',
        35: '_background_noise_'}

    elif nWords == 21:
     words = {
        '_background_noise_': 20,
        'down': 17,
        'go': 12,
        'left': 5,
        'no': 3,
        'off': 14,
        'on': 7,
        'right': 2,
        'stop': 15,
        'up': 4,
        'yes': 9,
        'zero': 19,
        'one': 8,
        'two': 13,
        'three': 1,
        'four': 10,
        'five': 16,
        'six': 6,
        'seven': 18,
        'eight': 11,
        'nine': 0
        }

     inv_words = {
        0: 'nine',
        1: 'three',
        2: 'right',
        3: 'no',
        4: 'up',
        5: 'left',
        6: 'six',
        7: 'on',
        8: 'one',
        9: 'yes',
        10: 'four',
        11: 'eight',
        12: 'go',
        13: 'two',
        14: 'off',
        15: 'stop',
        16: 'five',
        17: 'down',
        18: 'seven',
        19: 'zero',
        20: '_background_noise_'
        }
    # take the txt files for taking the correct files
    valWavs = open(data_path + 'train/validation_list.txt').read().splitlines()
    testWavs = open(data_path + 'train/testing_list.txt').read().splitlines()

    valWavs = ['dataset/train/'+f for f in valWavs]
    testWavs = ['dataset/train/'+f for f in testWavs]

    # Find trainWavs as allFiles
    allFiles = list()
    for root, dirs, files in os.walk(data_path+'train/'):
        allFiles += [root+'/' + f for f in files if f.endswith('.wav')]
    trainWavs = list(set(allFiles)-set(valWavs)-set(testWavs))

    # Get labels
    valWavLabels = [getLabel(wav) for wav in valWavs]
    testWavLabels = [getLabel(wav) for wav in testWavs]
    trainWavLabels = [getLabel(wav) for wav in trainWavs]

    # Create dictionaries containing related labels
    trainData = {'files': trainWavs, 'labels': trainWavLabels}
    valData = {'files': valWavs, 'labels': valWavLabels}
    testData = {'files': testWavs, 'labels': testWavLabels}

    dataDict = {
        'train': trainData,
        'val': valData,
        'test': testData,
        }

    return dataDict


def downloadFile(url, file_name):
    """
    This function gets file from url
    :param url: url for dataset
    :param file_name: takes folder names such as 'go' or 'down'
    :return: data
    """
    data_request = requests.get(url)
    print('Downloading {} into {}'.format(url, file_name))
    with open(file_name, 'wb') as f:
        f.write(data_request.content)


def extractFile(file_name, directory):
    """

    :param file_name: audio files
    :param directory: extraction directory
    :return: extracts the files to the directory
    """
    print('Extracting {} into {}'.format(file_name, directory))
    if (file_name.endswith("tar.gz")):
        tar = tarfile.open(file_name, "r:gz")
        tar.extractall(path=directory)
        tar.close()
    else:
        print('Unknown format.')


def getLabel(file_name):
    """
    Get the label from file path
    path = */baseDir/train/CATEGORY/file_name
    """
    category = file_name.split('/')[-2]
    return words.get(category, words['_background_noise_'])


def getDataframe(data, include_unknown=False):
    """
    Create a dataframe from a Dictionary and remove _background_noise_
    """
    df = pd.DataFrame(data)
    df['category'] = df.apply(
        lambda row: inv_words[row['labels']], axis=1
        )
    if not include_unknown:
        df = df.loc[df['category'] != '_background_noise_', :]

    return df
