# Number of classes including _background_noise_
NUM_CLASSES = 36

# Sampling rate
AUDIO_SR = 16000

# Length of a single wav
AUDIO_LENGTH = 16000

# Length of a single wav for Librosa
LIBROSA_AUDIO_LENGTH = 22050

categories = {
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
    'visual': 34,

    }

inv_categories = {
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
    35: '_background_noise_'
    }

# # Number of classes including _background_noise_
# NUM_CLASSES = 10

# # Sampling rate
# AUDIO_SR = 16000

# # Length of a single wav
# AUDIO_LENGTH = 16000

# # Length of a single wav for Librosa
# LIBROSA_AUDIO_LENGTH = 22050

# categories = {
#     '_background_noise_': 10,
#     'down': 6,
#     'go': 8,
#     'left': 5,
#     'no': 3,
#     'off': 1,
#     'on': 4,
#     'right': 2,
#     'stop': 0,
#     'up': 7,
#     'yes': 9
#     }

# inv_categories = {
#     0: 'stop',
#     1: 'off',
#     2: 'right',
#     3: 'no',
#     4: 'on',
#     5: 'left',
#     6: 'down',
#     7: 'up',
#     8: 'go',
#     9: 'yes',
#     10: '_background_noise_'
#     }

# # Number of classes including _background_noise_
# NUM_CLASSES = 21

# # Sampling rate
# AUDIO_SR = 16000

# # Length of a single wav
# AUDIO_LENGTH = 16000

# # Length of a single wav for Librosa
# LIBROSA_AUDIO_LENGTH = 22050

# categories = {
#     '_background_noise_': 20,
#     'down': 17,
#     'go': 12,
#     'left': 5,
#     'no': 3,
#     'off': 14,
#     'on': 7,
#     'right': 2,
#     'stop': 15,
#     'up': 4,
#     'yes': 9,
#     'zero': 19,
#     'one': 8,
#     'two': 13,
#     'three': 1,
#     'four': 10,
#     'five': 16,
#     'six': 6,
#     'seven': 18,
#     'eight': 11,
#     'nine': 0
#     }

# inv_categories = {
#     0: 'nine',
#     1: 'three',
#     2: 'right',
#     3: 'no',
#     4: 'up',
#     5: 'left',
#     6: 'six',
#     7: 'on',
#     8: 'one',
#     9: 'yes',
#     10: 'four',
#     11: 'eight',
#     12: 'go',
#     13: 'two',
#     14: 'off',
#     15: 'stop',
#     16: 'five',
#     17: 'down',
#     18: 'seven',
#     19: 'zero',
#     20: '_background_noise_'
#     }
