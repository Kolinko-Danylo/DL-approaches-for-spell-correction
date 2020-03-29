import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import nlpaug.augmenter.char as nac
import fasttext
import spacy
import gensim
from gensim.models.wrappers import FastText
from torch.utils import data



class DataLoader(data.Dataset):
    def __init__(self, root_path, seq_length):
        """Initialize the dataset"""
        self.chars = None
        self.root_path = root_path
        self.seq_legth = seq_length
        self.tokenized_data = self.tokenize(root_path)
        self.tokenized_data = self.tokenized_data[:len(self.tokenized_data)//self.seq_legth]
        self.data = np.array(self.tokenized_data).reshape(-1, seq_length)


        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        # encoded = np.array([char2int[ch] for ch in text])
        self.words = tuple(set(self.tokenized_data))
        self.int2word = dict(enumerate(words))
        self.word2int = {ch: ii for ii, ch in self.int2word.items()}


    def tokenize(self, root_path):
        with open(root_path, 'r') as f:
            text = f.read()
            self.chars = tuple(set(text))

        spacy_nlp = spacy.load('en_core_web_sm')
        spacy_nlp.max_length = 2 * len(text)

        x = spacy_nlp(text, disable=['parser', 'tagger', 'ner'])
        splitted = [token.text for token in x if
                    not (token.text.isspace() or (token.text[0].isdigit() and token.text[-1].isdigit()))]
        return splitted

    def one_hot_encode(self, arr, n_labels):
        one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)

        # Fill the appropriate elements with ones
        one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

        # Finally reshape it to get back to the original array
        one_hot = one_hot.reshape((*arr.shape, n_labels))

        return one_hot

    def middle_embedding(self, word, n_labels):
        return np.sum(self.one_hot_encode(word[1:-1], n_labels), axis=0)

    def augment(self, arr):
        augmentator = nac.KeyboardAug(aug_char_min=0, aug_char_p=0.4, aug_word_p=0.4, aug_word_min=0,
                                 aug_word_max=arr.size // 3, special_char=False)
        augmented_data = augmentator.augment(" ".join(arr.ravel().tolist())).split()
        return np.array(augmented_data).reshape(arr.shape)

    def get_encodes(self, arr):
        arr = self.augment(arr).ravel()

        flat_arr = arr
        splitted_encoded = np.array(list(map(lambda x: np.array([self.char2int[ch] for ch in x]), flat_arr)))

        first_char = list(map(lambda x: x[0], splitted_encoded))
        last_char = list(map(lambda x: x[-1], splitted_encoded))
        middle = splitted_encoded.tolist()

        first_char_encoded = self.one_hot_encode(np.array(first_char), len(self.chars))
        last_char_encoded = self.one_hot_encode(np.array(last_char), len(self.chars))

        middle_encoded = np.vstack(list(map(lambda x: self.middle_embedding(x, len(self.chars)), middle)))
        encoded_seq = np.hstack([first_char_encoded, middle_encoded, last_char_encoded]).reshape(
            (*arr.shape, 3 * len(self.chars)))
        return encoded_seq

    def get_int(self, x):
        return self.word2int[x]

    def get_params(self):
      param_dict = dict()

      param_dict["chars"] = self.chars
      param_dict["int2char"] = self.int2char
      param_dict["char2int"] = self.char2int
      param_dict["words"] = self.words
      param_dict["int2word"] = self.int2word
      param_dict["word2int"] = self.word2int

      return param_dict

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, index):
        sequence = self.data[index]
        X = self.get_encodes(sequence.copy())
        y = np.vectorize(self.get_int)(sequence)
        return torch.from_numpy(X), torch.from_numpy(y)


# def get_batches(arr, batch_size, seq_length):
#     '''Create a generator that returns batches of size
#        batch_size x seq_length from arr.
#
#        Arguments
#        ---------
#        arr: Array you want to make batches from
#        batch_size: Batch size, the number of sequences per batch
#        seq_length: Number of encoded chars in a sequence
#     '''
#
#     batch_size_total = batch_size * seq_length
#     # total number of batches we can make
#     n_batches = len(arr) // batch_size_total
#
#     # Keep only enough characters to make full batches
#     arr = arr[:n_batches * batch_size_total]
#
#     # Reshape into batch_size rows
#     arr = arr.reshape((batch_size, -1))
#     # iterate through the array, one sequence at a time
#     for n in range(0, arr.shape[1], seq_length):
#         # The features
#         base = arr[:, n:n + seq_length]
#         y = np.vectorize(emb.get_int)(base)
#
#         x = base.copy()
#         x = emb.get_encodes(x, use_aug=True)
#
#         yield x, y




