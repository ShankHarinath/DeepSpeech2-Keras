"""
Defines a class that is used to featurize audio clips, and provide
them to the network for training or testing.
"""

from __future__ import absolute_import, division, print_function

import os
import random
import wave
from concurrent.futures import ThreadPoolExecutor, wait
from functools import reduce

import numpy as np
import soundfile
from numpy.lib.stride_tricks import as_strided

RNG_SEED = 123
char_map_str = """
' 1
<SPACE> 2
a 3
b 4
c 5
d 6
e 7
f 8
g 9
h 10
i 11
j 12
k 13
l 14
m 15
n 16
o 17
p 18
q 19
r 20
s 21
t 22
u 23
v 24
w 25
x 26
y 27
z 28
"""
char_map = {}
index_map = {}
for line in char_map_str.strip().split('\n'):
    ch, index = line.split()
    char_map[ch] = int(index)
    index_map[int(index)] = ch
index_map[2] = ' '


def calc_feat_dim(window, max_freq):
    return int(0.001 * window * max_freq) + 1


def text_to_int_sequence(text):
    """ Use a character map and convert text to an integer sequence """
    int_sequence = []
    for c in text:
        if c == ' ':
            ch = char_map['<SPACE>']
        else:
            ch = char_map[c]
        int_sequence.append(ch)
    return int_sequence


def int_to_text_sequence(seq):
    text_sequence = []
    for c in seq:
        if c == 28:
            ch = ''
        else:
            ch = index_map[c]
        text_sequence.append(ch)
    return text_sequence


def spectrogram(samples, fft_length=256, sample_rate=2, hop_length=128):
    """
    Compute the spectrogram for a real signal.
    The parameters follow the naming convention of
    matplotlib.mlab.specgram
    Args:
        samples (1D array): input audio signal
        fft_length (int): number of elements in fft window
        sample_rate (scalar): sample rate
        hop_length (int): hop length (relative offset between neighboring
            fft windows).
    Returns:
        x (2D array): spectrogram [frequency x time]
        freq (1D array): frequency of each row in x
    Note:
        This is a truncating computation e.g. if fft_length=10,
        hop_length=5 and the signal has 23 elements, then the
        last 3 elements will be truncated.
    """
    assert not np.iscomplexobj(samples), "Must not pass in complex numbers"

    window = np.hanning(fft_length)[:, None]
    window_norm = np.sum(window ** 2)

    # The scaling below follows the convention of
    # matplotlib.mlab.specgram which is the same as
    # matlabs specgram.
    scale = window_norm * sample_rate

    trunc = (len(samples) - fft_length) % hop_length
    x = samples[:len(samples) - trunc]

    # "stride trick" reshape to include overlap
    nshape = (fft_length, (len(x) - fft_length) // hop_length + 1)
    nstrides = (x.strides[0], x.strides[0] * hop_length)
    x = as_strided(x, shape=nshape, strides=nstrides)

    # window stride sanity check
    assert np.all(x[:, 1] == samples[hop_length:(hop_length + fft_length)])

    # broadcast window, compute fft over columns and square mod
    x = np.fft.rfft(x * window, axis=0)
    x = np.absolute(x) ** 2

    # scale, 2.0 for everything except dc and fft_length/2
    x[1:-1, :] *= (2.0 / scale)
    x[(0, -1), :] /= scale

    freqs = float(sample_rate) / fft_length * np.arange(x.shape[0])

    return x, freqs


def spectrogram_from_file(filename, step=10, window=20, max_freq=None,
                          eps=1e-14):
    """ Calculate the log of linear spectrogram from FFT energy
    Params:
        filename (str): Path to the audio file
        step (int): Step size in milliseconds between windows
        window (int): FFT window size in milliseconds
        max_freq (int): Only FFT bins corresponding to frequencies between
            [0, max_freq] are returned
        eps (float): Small value to ensure numerical stability (for ln(x))
    """
    with soundfile.SoundFile(filename) as sound_file:
        audio = sound_file.read(dtype='float32')
        sample_rate = sound_file.samplerate
        if audio.ndim >= 2:
            audio = np.mean(audio, 1)
        if max_freq is None:
            max_freq = sample_rate / 2
        if max_freq > sample_rate / 2:
            raise ValueError("max_freq must not be greater than half of "
                             " sample rate")
        if step > window:
            raise ValueError("step size must not be greater than window size")
        hop_length = int(0.001 * step * sample_rate)
        fft_length = int(0.001 * window * sample_rate)
        pxx, freqs = spectrogram(
            audio, fft_length=fft_length, sample_rate=sample_rate,
            hop_length=hop_length)
        ind = np.where(freqs <= max_freq)[0][-1] + 1
    return np.transpose(np.log(pxx[:ind, :] + eps))


class DataGenerator(object):
    def __init__(self, step=10, window=20, max_freq=8000, desc_file=None):
        """
        Params:
            step (int): Step size in milliseconds between windows
            window (int): FFT window size in milliseconds
            max_freq (int): Only FFT bins corresponding to frequencies between
                [0, max_freq] are returned
            desc_file (str, optional): Path to a JSON-line file that contains
                labels and paths to the audio files. If this is None, then
                load metadata right away
        """
        self.feat_dim = calc_feat_dim(window, max_freq)
        self.feats_mean = np.zeros((self.feat_dim,))
        self.feats_std = np.ones((self.feat_dim,))
        self.rng = random.Random(RNG_SEED)
        if desc_file is not None:
            self.load_metadata_from_desc_file(desc_file)
        self.step = step
        self.window = window
        self.max_freq = max_freq

    def read_data(self, data_directory, max_duration=10.0):
        labels = []
        durations = []
        keys = []
        for group in os.listdir(data_directory):
            speaker_path = os.path.join(data_directory, group)
            if not os.path.isdir(speaker_path):
                continue
            for speaker in os.listdir(speaker_path):
                chapter_path = os.path.join(speaker_path, speaker)
                if not os.path.isdir(chapter_path):
                    continue
                for chapter in os.listdir(chapter_path):
                    labels_file = os.path.join(chapter_path, chapter,
                                               '{}-{}.trans.txt'
                                               .format(speaker, chapter))
                    for line in open(labels_file):
                        split = line.strip().split()
                        file_id = split[0]
                        label = ' '.join(split[1:]).lower()
                        audio_file = os.path.join(chapter_path, chapter,
                                                  file_id) + '.wav'
                        audio = wave.open(audio_file)
                        duration = float(audio.getnframes()) / audio.getframerate()
                        audio.close()
                        if float(duration) > max_duration:
                            continue
                        keys.append(audio_file)
                        durations.append(duration)
                        labels.append(label)

        return keys, durations, labels

    def featurize(self, audio_clip):
        """ For a given audio clip, calculate the log of its Fourier Transform
        Params:
            audio_clip(str): Path to the audio clip
        """
        return spectrogram_from_file(
            audio_clip, step=self.step, window=self.window,
            max_freq=self.max_freq)

    def load_data(self, data_directory, partition='train',
                  max_duration=10.0):
        """ Read metadata from the description file
            (possibly takes long, depending on the filesize)
        Params:
            desc_file (str):  Path to a JSON-line file that contains labels and
                paths to the audio files
            partition (str): One of 'train', 'validation' or 'test'
            max_duration (float): In seconds, the maximum duration of
                utterances to train or test on
        """
        audio_paths, durations, texts = self.read_data(data_directory, max_duration)
        if partition == 'train':
            self.train_audio_paths = audio_paths
            self.train_durations = durations
            self.train_texts = texts
        elif partition == 'validation':
            self.val_audio_paths = audio_paths
            self.val_durations = durations
            self.val_texts = texts
        elif partition == 'test':
            self.test_audio_paths = audio_paths
            self.test_durations = durations
            self.test_texts = texts
        else:
            raise Exception("Invalid partition to load metadata. "
                            "Must be train/validation/test")

    def load_train_data(self, data_directory):
        self.load_data(data_directory, 'train')

    def load_test_data(self, data_directory):
        self.load_data(data_directory, 'test')

    def load_validation_data(self, data_directory):
        self.load_data(data_directory, 'validation')

    @staticmethod
    def sort_by_duration(durations, audio_paths, texts):
        return zip(*sorted(zip(durations, audio_paths, texts)))

    def normalize(self, feature, eps=1e-14):
        return (feature - self.feats_mean) / (self.feats_std + eps)

    def prepare_batch(self, audio_paths, texts):
        """ Featurize a batch of audio, zero pad them and return a dictionary
        Params:
            audio_paths (list(str)): List of paths to audio files
            texts (list(str)): List of texts corresponding to the audio files
        Returns:
            dict: See below for contents
        """
        assert len(audio_paths) == len(texts), \
            "Inputs and outputs to the network must be of the same number"
        # Features is a list of (timesteps, feature_dim) arrays
        # Calculate the features for each audio clip, as the log of the
        # Fourier Transform of the audio
        features = [self.featurize(a) for a in audio_paths]
        input_lengths = [f.shape[0] for f in features]
        max_length = max(input_lengths)
        feature_dim = features[0].shape[1]
        mb_size = len(features)
        # Pad all the inputs so that they are all the same length
        x = np.zeros((mb_size, max_length, feature_dim))
        y = []
        label_lengths = []
        for i in range(mb_size):
            feat = features[i]
            feat = self.normalize(feat)  # Center using means and std
            x[i, :feat.shape[0], :] = feat
            label = text_to_int_sequence(texts[i])
            y.append(label)
            label_lengths.append(len(label))
        # Flatten labels to comply with warp-CTC signature
        y = reduce(lambda i, j: i + j, y)
        return {
            'x': x,  # (0-padded features of shape(batch_size, timesteps, feat_dim)
            'y': y,  # list(int) Flattened labels (integer sequences)
            'texts': texts,  # list(str) Original texts
            'input_lengths': input_lengths,  # list(int) Length of each input
            'label_lengths': label_lengths  # list(int) Length of each label
        }

    def get_generator(self, audio_paths, texts, batch_size, shuffle=True, sort_by_duration=False):
        def generator():
            num_samples = len(audio_paths)
            while True:
                if shuffle:
                    temp = list(zip(audio_paths, texts))
                    self.rng.shuffle(temp)
                    x, y = list(zip(*temp))

                pool = ThreadPoolExecutor(1)  # Run a single I/O thread in parallel
                future = pool.submit(self.prepare_batch,
                                     x[:batch_size],
                                     y[:batch_size])
                for offset in range(batch_size, num_samples, batch_size):
                    wait([future])
                    batch = future.result()
                    future = pool.submit(self.prepare_batch,
                                         x[offset: offset + batch_size],
                                         y[offset: offset + batch_size])
                    yield batch

        return generator()

    def get_train_generator(self, batch_size=16, shuffle=True):
        return self.get_generator(self.train_audio_paths, self.train_texts, batch_size, shuffle)

    def get_test_generator(self, batch_size=16, shuffle=True):
        return self.get_generator(self.test_audio_paths, self.test_texts, batch_size, shuffle)

    def get_validation_generator(self, batch_size=16, shuffle=True):
        return self.get_generator(self.val_audio_paths, self.val_texts, batch_size, shuffle)

    def fit_train(self, k_samples=100):
        """ Estimate the mean and std of the features from the training set
        Params:
            k_samples (int): Use this number of samples for estimation
        """
        k_samples = min(k_samples, len(self.train_audio_paths))
        samples = self.rng.sample(self.train_audio_paths, k_samples)
        feats = [self.featurize(s) for s in samples]
        feats = np.vstack(feats)
        self.feats_mean = np.mean(feats, axis=0)
        self.feats_std = np.std(feats, axis=0)

