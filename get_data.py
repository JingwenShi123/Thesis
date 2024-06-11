"""Implements dataloaders for AFFECT data."""
import os
import sys
from typing import *
import pickle
import h5py
from torch.nn import functional as F
sys.path.append(os.getcwd())
import torch
import torchtext as text
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import numpy as np
import re
from tqdm import tqdm
from torch.utils.data import ConcatDataset, DataLoader

def add_text_noise(tests, noise_level=0.3, swap=True, rand_mid=True, typo=True, sticky=True, omit=True):
    """
    Add various types of noise to text data.

    :param noise_level: Probability of randomly applying noise to a word. ( default: 0.1)
    :param swap:  Swap two adjacent letters. ( default: True )
    :param rand_mid: Randomly permute the middle section of the word, except for the first and last letters. ( default: True )
    :param typo: Simulate keyboard typos for the word. ( default: True )
    :param sticky: Randomly repeat letters inside a word. ( default: True )
    :param omit: Randomly omit some letters from a word ( default: True )
    """
    noises = []
    if swap:
        noises.append(swap_letter)
    if rand_mid:
        noises.append(random_mid)
    if typo:
        noises.append(qwerty_typo)
    if sticky:
        noises.append(sticky_keys)
    if omit:
        noises.append(omission)
    robustness_tests = []
    for i in tqdm(range(len(tests))):
        newtext = []
        text = _normalizeText(tests[i])
        for word in text:
            if _last_char(word) > 3 and np.random.sample() <= noise_level:
                mode = np.random.randint(len(noises))
                newtext.append(noises[mode](word))
            else:
                newtext.append(word)
        robustness_tests.append(' '.join(newtext))
    return robustness_tests


def _normalizeText(text):
    """Normalize text before transforming."""
    text = text.lower()
    text = re.sub(r'<br />', r' ', text).strip()
    text = re.sub(r'^https?:\/\/.*[\r\n]*', ' L ', text, flags=re.MULTILINE)
    text = re.sub(r'[\~\*\+\^`_#\[\]|]', r' ', text).strip()
    text = re.sub(r'[0-9]+', r' N ', text).strip()
    text = re.sub(r'([/\'\-\.?!\(\)",:;])', r' \1 ', text).strip()
    return text.split()


def _last_char(word):
    """Get last alphanumeric character of word.

    :param word: word to get the last letter of.
    """
    for i in range(len(word)):
        if word[len(word) - 1 - i].isalpha() or word[len(word) - 1 - i].isdigit():
            return len(word) - 1 - i
    return -1


def swap_letter(word):
    """Swap two random adjacent letters.

    :param word: word to apply transformations to.
    """
    last = _last_char(word)
    pos = np.random.randint(last - 2) + 1
    return word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]


def random_mid(word):
    """Randomly permute the middle chunk of a word (all letters except the first and last letter).

    :param word: word to apply transformations to.
    """
    last = _last_char(word)
    mid = [char for char in word[1:last]]
    np.random.shuffle(mid)
    return word[0] + ''.join(mid) + word[last:]


def qwerty_typo(word):
    """Randomly replace num_typo number of letters of a word to a one adjacent to it on qwerty keyboard.

    :param word: word to apply transformations to.:
    """
    qwerty = {'q': ['w'], 'w': ['q', 'e', 's'], 'e': ['w', 'r', 'd'], 'r': ['e', 't', 'f'], 't': ['r', 'g', 'y'],
              'y': ['t', 'u', 'h'], 'u': ['y', 'i', 'j'], 'i': ['u', 'o', 'k'], 'o': ['i', 'p', 'l'], 'p': ['o'],
              'a': ['q', 's', 'z'], 's': ['a', 'w', 'd', 'x', 'z'], 'd': ['s', 'e', 'f', 'x', 'c'],
              'f': ['d', 'r', 'g', 'c', 'v'], 'g': [
            'f', 't', 'h', 'v', 'b'], 'h': ['g', 'y', 'j', 'b', 'n'], 'j': ['h', 'u', 'k', 'n', 'm'],
              'k': ['j', 'i', 'l', 'm'], 'l': ['k', 'o'], 'z': ['a', 's', 'x'], 'x': ['z', 's', 'd', 'c'],
              'c': ['x', 'd', 'f', 'v'], 'v': ['c', 'f', 'g', 'b'], 'b': ['v', 'g', 'h', 'n'],
              'n': ['b', 'h', 'm', 'j'], 'm': ['n', 'j', 'k']}
    last = _last_char(word)
    typos = np.arange(last + 1)
    np.random.shuffle(typos)
    for i in range(len(typos)):
        if word[typos[i]] in qwerty:
            typo = qwerty[word[typos[i]]]
            key = typo[np.random.randint(len(typo))]
            word = word[:typos[i]] + key + word[typos[i] + 1:]
            break
    return word


def sticky_keys(word, num_sticky=1):
    """Randomly repeat letters of a word once.

    :param word: word to apply transformations to.
    :param num_sticky: Number of letters to randomly repeat once.
    """
    last = _last_char(word)
    sticky = np.arange(last + 1)
    np.random.shuffle(sticky)
    for i in range(num_sticky):
        word = word[:sticky[i]] + word[sticky[i]] + word[sticky[i]:]
    return word


def omission(word, num_omit=1):
    """Randomly omit num_omit number of letters of a word.

    :param word: word to apply transformations to.
    :param num_sticky: Number of letters to randomly omit.
    """
    last = _last_char(word)
    for i in range(num_omit):
        omit = np.random.randint(last - 1) + 1
        word = word[:omit] + word[omit + 1:]
        last -= 1
    return word

#######################################################################
# Time-Series
def add_timeseries_noise(tests, noise_level=0.3, gaussian_noise=True, rand_drop=True, struct_drop=True):
    """
    Add various types of noise to timeseries data.

    :param noise_level: Standard deviation of gaussian noise, and drop probability in random drop and structural drop
    :param gauss_noise:  Add Gaussian noise to the time series ( default: True )
    :param rand_drop: Add randomized dropout to the time series ( default: True )
    :param struct_drop: Add randomized structural dropout to the time series ( default: True )
    """
    # robust_tests = np.array(tests)
    robust_tests = tests
    if gaussian_noise:
        robust_tests = white_noise(robust_tests, noise_level)
    if rand_drop:
        robust_tests = random_drop(robust_tests, noise_level)
    if struct_drop:
        robust_tests = structured_drop(robust_tests, noise_level)
    return robust_tests


def white_noise(data, p):
    """Add noise sampled from zero-mean Gaussian with standard deviation p at every time step.

    :param data: Data to process.
    :param p: Standard deviation of added Gaussian noise.
    """
    for i in range(len(data)):
        for time in range(len(data[i])):
            data[i][time] += np.random.normal(0, p)
    return data


def random_drop(data, p):
    """Drop each time series entry independently with probability p.

    :param data: Data to process.
    :param p: Probability to drop feature.
    """
    for i in range(len(data)):
        data[i] = _random_drop_helper(data[i], p, len(np.array(data).shape))
    return data


def _random_drop_helper(data, p, level):
    """
    Helper function that implements random drop for 2-/higher-dimentional timeseris data.

    :param data: Data to process.
    :param p: Probability to drop feature.
    :param level: Dimensionality.
    """
    if level == 2:
        for i in range(len(data)):
            if np.random.random_sample() < p:
                data[i] = 0
        return data
    else:
        for i in range(len(data)):
            data[i] = _random_drop_helper(data[i], p, level - 1)
        return data


def structured_drop(data, p):
    """Drop each time series entry independently with probability p, but drop all modalities if you drop an element.

    :param data: Data to process.
    :param p: Probability to drop entire element of time series.
    """
    for i in range(len(data)):
        for time in range(len(data[i])):
            if np.random.random_sample() < p:
                data[i][time] = np.zeros(data[i][time].shape)
    return data

#######################################################################

np.seterr(divide='ignore', invalid='ignore')

def drop_entry(dataset):
    """Drop entries where there's no text in the data."""
    drop = []
    for ind, k in enumerate(dataset["text"]):
        if k.sum() == 0:
            drop.append(ind)
    for modality in list(dataset.keys()):
        dataset[modality] = np.delete(dataset[modality], drop, 0)
    return dataset


def z_norm(dataset, max_seq_len=50):
    """Normalize data in the dataset."""
    processed = {}
    text = dataset['text'][:, :max_seq_len, :]
    vision = dataset['vision'][:, :max_seq_len, :]
    audio = dataset['audio'][:, :max_seq_len, :]
    for ind in range(dataset["text"].shape[0]):
        vision[ind] = np.nan_to_num(
            (vision[ind] - vision[ind].mean(0, keepdims=True)) / (np.std(vision[ind], axis=0, keepdims=True)))
        audio[ind] = np.nan_to_num(
            (audio[ind] - audio[ind].mean(0, keepdims=True)) / (np.std(audio[ind], axis=0, keepdims=True)))
        text[ind] = np.nan_to_num(
            (text[ind] - text[ind].mean(0, keepdims=True)) / (np.std(text[ind], axis=0, keepdims=True)))

    processed['vision'] = vision
    processed['audio'] = audio
    processed['text'] = text
    processed['labels'] = dataset['labels']
    return processed


def get_rawtext(path, data_kind, vids):
    """Get raw text, video data from hdf5 file."""
    if data_kind == 'hdf5':
        f = h5py.File(path, 'r')
    else:
        with open(path, 'rb') as f_r:
            f = pickle.load(f_r)
    text_data = []
    new_vids = []

    for vid in vids:
        text = []
        # If data IDs are NOT the same as the raw ids
        # add some code to match them here, eg. from vanvan_10 to vanvan[10]
        # (id, seg) = re.match(r'([-\w]*)_(\w+)', vid).groups()
        # vid_id = '{}[{}]'.format(id, seg)
        vid_id = int(vid[0]) if type(vid) == np.ndarray else vid
        try:
            if data_kind == 'hdf5':
                for word in f['words'][vid_id]['features']:
                    if word[0] != b'sp':
                        text.append(word[0].decode('utf-8'))
                text_data.append(' '.join(text))
                new_vids.append(vid_id)
            else:
                for word in f[vid_id]:
                    if word != 'sp':
                        text.append(word)
                text_data.append(' '.join(text))
                new_vids.append(vid_id)
        except:
            print("missing", vid, vid_id)
    return text_data, new_vids


def _get_word2id(text_data, vids):
    word2id = defaultdict(lambda: len(word2id))
    UNK = word2id['unk']
    data_processed = dict()
    for i, segment in enumerate(text_data):
        words = []
        _words = segment.split()
        for word in _words:
            words.append(word2id[word])
        words = np.asarray(words)
        data_processed[vids[i]] = words

    def _return_unk():
        return UNK

    word2id.default_factory = _return_unk
    return data_processed, word2id


def _get_word_embeddings(word2id, save=False):
    vec = text.vocab.GloVe(name='840B', dim=300)
    tokens = []
    for w, _ in word2id.items():
        tokens.append(w)

    ret = vec.get_vecs_by_tokens(tokens, lower_case_backup=True)
    return ret


def _glove_embeddings(text_data, vids, paddings=50):
    data_prod, w2id = _get_word2id(text_data, vids)
    word_embeddings_looks_up = _get_word_embeddings(w2id)
    looks_up = word_embeddings_looks_up.numpy()
    embedd_data = []
    for vid in vids:
        d = data_prod[vid]
        tmp = []
        look_up = [looks_up[x] for x in d]
        # Padding with zeros at the front
        # TODO: fix some segs have more than 50 words (FIXed)
        if len(d) > paddings:
            for x in d[:paddings]:
                tmp.append(looks_up[x])
        else:
            for i in range(paddings - len(d)):
                tmp.append(np.zeros(300, ))
            for x in d:
                tmp.append(looks_up[x])
        # try:
        #     tmp = [looks_up[x] for x in d]
        # except:

        embedd_data.append(np.array(tmp))
    return np.array(embedd_data)


class Affectdataset(Dataset):
    """Implements Affect data as a torch dataset."""
    def __init__(self, data: Dict, flatten_time_series: bool, aligned: bool = True, task: str = None, max_pad=False, max_pad_num=50, data_type='mosi', z_norm=False) -> None:
        """Instantiate AffectDataset

        Args:
            data (Dict): Data dictionary
            flatten_time_series (bool): Whether to flatten time series or not
            aligned (bool, optional): Whether to align data or not across modalities. Defaults to True.
            task (str, optional): What task to load. Defaults to None.
            max_pad (bool, optional): Whether to pad data to max_pad_num or not. Defaults to False.
            max_pad_num (int, optional): Maximum padding number. Defaults to 50.
            data_type (str, optional): What data to load. Defaults to 'mosi'.
            z_norm (bool, optional): Whether to normalize data along the z-axis. Defaults to False.
        """
        self.dataset = data
        self.flatten = flatten_time_series
        self.aligned = aligned
        self.task = task
        self.max_pad = max_pad
        self.max_pad_num = max_pad_num
        self.data_type = data_type
        self.z_norm = z_norm
        self.dataset['audio'][self.dataset['audio'] == -np.inf] = 0.0

    def __getitem__(self, ind):
        """Get item from dataset."""
        # vision = torch.tensor(vision)
        # audio = torch.tensor(audio)
        # text = torch.tensor(text)

        vision = torch.tensor(self.dataset['vision'][ind])
        audio = torch.tensor(self.dataset['audio'][ind])
        text = torch.tensor(self.dataset['text'][ind])





        if self.aligned:
            try:
                start = text.nonzero(as_tuple=False)[0][0]
                # start = 0
            except:
                print(text, ind)
                exit()
            vision = vision[start:].float()
            audio = audio[start:].float()
            text = text[start:].float()
        else:
            vision = vision[vision.nonzero()[0][0]:].float()
            audio = audio[audio.nonzero()[0][0]:].float()
            text = text[text.nonzero()[0][0]:].float()

        # z-normalize data
        if self.z_norm:
            vision = torch.nan_to_num((vision - vision.mean(0, keepdims=True)) / (torch.std(vision, axis=0, keepdims=True)))
            audio = torch.nan_to_num((audio - audio.mean(0, keepdims=True)) / (torch.std(audio, axis=0, keepdims=True)))
            text = torch.nan_to_num((text - text.mean(0, keepdims=True)) / (torch.std(text, axis=0, keepdims=True)))

        def _get_class(flag, data_type=self.data_type):
            if data_type in ['mosi', 'mosei', 'sarcasm']:
                if flag > 0:
                    return [[1]]
                else:
                    return [[0]]
            else:
                return [flag]
        tmp_label = self.dataset['labels'][ind]
        if self.data_type == 'humor' or self.data_type == 'sarcasm':
            if (self.task == None) or (self.task == 'regression'):
                if self.dataset['labels'][ind] < 1:
                    tmp_label = [[-1]]
                else:
                    tmp_label = [[1]]
        else:
            tmp_label = self.dataset['labels'][ind]

        label = torch.tensor(_get_class(tmp_label)).long() if self.task == "classification" else torch.tensor(
            tmp_label).float()

        if self.flatten:
            return [vision.flatten(), audio.flatten(), text.flatten(), ind, \
                    label]
        else:
            if self.max_pad:
                tmp = [vision, audio, text, label]
                for i in range(len(tmp) - 1):
                    tmp[i] = tmp[i][:self.max_pad_num]
                    tmp[i] = F.pad(tmp[i], (0, 0, 0, self.max_pad_num - tmp[i].shape[0]))
            else:
                tmp = [vision, audio, text, ind, label]
            return tmp

    def __len__(self):
        """Get length of dataset."""
        return self.dataset['vision'].shape[0]


def get_dataloader(
        filepath: str, batch_size: int = 32, max_seq_len=50, max_pad=False, train_shuffle: bool = True,
        num_workers: int = 2, flatten_time_series: bool = False, task=None, robust_test=False, data_type='mosi',
        raw_path='/home/van/backup/pack/mosi/mosi.hdf5', z_norm=False) -> DataLoader:
    """Get dataloaders for affect data.

    Args:
        filepath (str): Path to datafile
        batch_size (int, optional): Batch size. Defaults to 32.
        max_seq_len (int, optional): Maximum sequence length. Defaults to 50.
        max_pad (bool, optional): Whether to pad data to max length or not. Defaults to False.
        train_shuffle (bool, optional): Whether to shuffle training data or not. Defaults to True.
        num_workers (int, optional): Number of workers. Defaults to 2.
        flatten_time_series (bool, optional): Whether to flatten time series data or not. Defaults to False.
        task (str, optional): Which task to load in. Defaults to None.
        robust_test (bool, optional): Whether to apply robustness to data or not. Defaults to False.
        data_type (str, optional): What data to load in. Defaults to 'mosi'.
        raw_path (str, optional): Full path to data. Defaults to '/home/van/backup/pack/mosi/mosi.hdf5'.
        z_norm (bool, optional): Whether to normalize data along the z dimension or not. Defaults to False.

    Returns:
        DataLoader: tuple of train dataloader, validation dataloader, test dataloader
    """
    with open(filepath, "rb") as f:
        alldata = pickle.load(f)

    processed_dataset = {'train': {}, 'test': {}, 'valid': {}}
    alldata['train'] = drop_entry(alldata['train'])
    alldata['valid'] = drop_entry(alldata['valid'])
    alldata['test'] = drop_entry(alldata['test'])

    process = eval("_process_2") if max_pad else eval("_process_1")

    train = list(alldata.values())[0]
    valid = list(alldata.values())[1]
    test = list(alldata.values())[2]

    train_v = list(train.values())[0]
    train_v_1 = train_v[:, :, :35]
    valid_v = list(valid.values())[0]
    valid_v_1 = valid_v[:, :, :35]
    test_v = list(test.values())[0]
    test_v_1 = test_v[:, :, :35]

    # 重新构建 data1，并将切片后的值赋值回去
    new_data1 = {}
    new_train = {key: value for key, value in train.items()}
    new_train[list(train.keys())[0]] = train_v_1
    new_data1['train'] = new_train

    new_valid = {key: value for key, value in valid.items()}
    new_valid[list(valid.keys())[0]] = valid_v_1
    new_data1['valid'] = new_valid

    new_test = {key: value for key, value in test.items()}
    new_test[list(test.keys())[0]] = test_v_1
    new_data1['test'] = new_test
    alldata = new_data1

    for dataset in alldata:
        processed_dataset[dataset] = alldata[dataset]

    valid = DataLoader(Affectdataset(processed_dataset['valid'], flatten_time_series, task=task, max_pad=max_pad, max_pad_num=max_seq_len, data_type=data_type, z_norm=z_norm), \
                       shuffle=False, num_workers=num_workers, batch_size=batch_size, \
                       collate_fn=process)

    combined_dataset = ConcatDataset([
        Affectdataset(processed_dataset['train'], flatten_time_series, task=task, max_pad=max_pad,
                      max_pad_num=max_seq_len, data_type=data_type, z_norm=z_norm),
        Affectdataset(processed_dataset['valid'], flatten_time_series, task=task, max_pad=max_pad,
                      max_pad_num=max_seq_len, data_type=data_type, z_norm=z_norm),
        Affectdataset(processed_dataset['test'], flatten_time_series, task=task, max_pad=max_pad,
                      max_pad_num=max_seq_len, data_type=data_type, z_norm=z_norm)
    ])

    # 创建一个DataLoader
    train = DataLoader(
        combined_dataset,
        shuffle=train_shuffle,  # 这里可以选择是否打乱数据
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=process
    )


    if robust_test:
        vids = [id for id in alldata['test']['id']]

        file_type = raw_path.split('.')[-1]  # hdf5
        rawtext, vids = get_rawtext(raw_path, file_type, vids)

        # Add text noises
        robust_text = []
        robust_text_numpy = []
        for i in range(10):
            test = dict()
            test['vision'] = alldata['test']["vision"]
            test['audio'] = alldata['test']["audio"]
            test['text'] = _glove_embeddings(add_text_noise(rawtext, noise_level=i / 10), vids)
            test['labels'] = alldata['test']["labels"]
            test = drop_entry(test)

            robust_text_numpy.append(test['text'])

            robust_text.append(
                DataLoader(Affectdataset(test, flatten_time_series, task=task, max_pad=max_pad, max_pad_num=max_seq_len, data_type=data_type, z_norm=z_norm), shuffle=False, num_workers=num_workers,
                        batch_size=batch_size, collate_fn=process))

        # Add visual noises
        robust_vision = []
        for i in range(10):
            test = dict()
            test['vision'] = add_timeseries_noise([alldata['test']['vision'].copy()], noise_level=i / 10, rand_drop=False)[0]

            test['audio'] = alldata['test']["audio"].copy()
            test['text'] = alldata['test']['text'].copy()
            test['labels'] = alldata['test']["labels"]
            test = drop_entry(test)
            print('test entries: {}'.format(test['vision'].shape))

            robust_vision.append(
                DataLoader(Affectdataset(test, flatten_time_series, task=task, max_pad=max_pad, max_pad_num=max_seq_len, data_type=data_type, z_norm=z_norm), shuffle=False, num_workers=num_workers,
                        batch_size=batch_size, collate_fn=process))

        # Add audio noises
        robust_audio = []
        for i in range(10):
            test = dict()
            test['vision'] = alldata['test']["vision"].copy()
            test['audio'] = add_timeseries_noise([alldata['test']['audio'].copy()], noise_level=i / 10, rand_drop=False)[0]
            test['text'] = alldata['test']['text'].copy()
            test['labels'] = alldata['test']["labels"]
            test = drop_entry(test)
            print('test entries: {}'.format(test['vision'].shape))

            robust_audio.append(
                DataLoader(Affectdataset(test, flatten_time_series, task=task, max_pad=max_pad, max_pad_num=max_seq_len, data_type=data_type, z_norm=z_norm), shuffle=False, num_workers=num_workers,
                        batch_size=batch_size, collate_fn=process))

        # Add timeseries noises

        # for i, text in enumerate(robust_text_numpy):

        #     alldata_test = timeseries_robustness([alldata['test']['vision'], alldata['test']['audio'], text], noise_level=i/10)
        #     test.append(alldata_test)

        robust_timeseries = []
        # alldata['test'] = drop_entry(alldata['test'])
        for i in range(10):
            robust_timeseries_tmp = add_timeseries_noise(
                [alldata['test']['vision'].copy(), alldata['test']['audio'].copy(), alldata['test']['text'].copy()],
                noise_level=i / (10 * 3), rand_drop=False)

            test = dict()
            test['vision'] = robust_timeseries_tmp[0]
            test['audio'] = robust_timeseries_tmp[1]
            test['text'] = robust_timeseries_tmp[2]
            test['labels'] = alldata['test']['labels']
            test = drop_entry(test)
            print('test entries: {}'.format(test['vision'].shape))

            robust_timeseries.append(
                DataLoader(Affectdataset(test, flatten_time_series, task=task, max_pad=max_pad, max_pad_num=max_seq_len, data_type=data_type, z_norm=z_norm), shuffle=False, num_workers=num_workers,
                        batch_size=batch_size, collate_fn=process))
        test_robust_data = dict()
        test_robust_data['robust_text'] = robust_text
        test_robust_data['robust_vision'] = robust_vision
        test_robust_data['robust_audio'] = robust_audio
        test_robust_data['robust_timeseries'] = robust_timeseries
        return train, valid, test_robust_data
    else:
        # test = dict()
        test = DataLoader(Affectdataset(processed_dataset['test'], flatten_time_series, task=task, max_pad=max_pad, max_pad_num=max_seq_len, data_type=data_type, z_norm=z_norm), \
                      shuffle=False, num_workers=num_workers, batch_size=batch_size, \
                      collate_fn=process)
        return train, valid, test

def _process_1(inputs: List):
    processed_input = []
    processed_input_lengths = []
    inds = []
    labels = []

    for i in range(len(inputs[0]) - 2):
        feature = []
        for sample in inputs:
            feature.append(sample[i])
        processed_input_lengths.append(torch.as_tensor([v.size(0) for v in feature]))
        pad_seq = pad_sequence(feature, batch_first=True)
        processed_input.append(pad_seq)

    for sample in inputs:

        inds.append(sample[-2])
        # if len(sample[-2].shape) > 2:
        #     labels.append(torch.where(sample[-2][:, 1] == 1)[0])
        # else:
        if sample[-1].shape[1] > 1:
            labels.append(sample[-1].reshape(sample[-1].shape[1], sample[-1].shape[0])[0])
        else:
            labels.append(sample[-1])

    return processed_input, processed_input_lengths, \
           torch.tensor(inds).view(len(inputs), 1), torch.tensor(labels).view(len(inputs), 1)


def _process_2(inputs: List):
    processed_input = []
    processed_input_lengths = []
    labels = []

    for i in range(len(inputs[0]) - 1):
        feature = []
        for sample in inputs:
            feature.append(sample[i])
        processed_input_lengths.append(torch.as_tensor([v.size(0) for v in feature]))
        # pad_seq = pad_sequence(feature, batch_first=True)
        processed_input.append(torch.stack(feature))

    for sample in inputs:

        # if len(sample[-2].shape) > 2:
        #     labels.append(torch.where(sample[-2][:, 1] == 1)[0])
        # else:
        # print(sample[-1].shape)
        if sample[-1].shape[1] > 1:
            labels.append(sample[-1].reshape(sample[-1].shape[1], sample[-1].shape[0])[0])
        else:
            labels.append(sample[-1])

    return processed_input[0], processed_input[1], processed_input[2], torch.tensor(labels).view(len(inputs), 1)
