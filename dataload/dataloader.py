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
import string

from bpemb import BPEmb



class SemicharDataset(data.Dataset):
    def __init__(self, root_path, seq_length):
        """Initialize the dataset"""
        self.chars = None
        self.root_path = root_path
        self.seq_legth = seq_length
        self.tokenized_data = self.tokenize(root_path)
        self.tokenized_data = self.tokenized_data[:len(self.tokenized_data) - len(self.tokenized_data) % self.seq_legth]
        self.data = np.array(self.tokenized_data).reshape(-1, seq_length)


        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        self.words = tuple(set(self.tokenized_data))
        self.int2word = dict(enumerate(self.words))
        self.word2int = {ch: ii for ii, ch in self.int2word.items()}


    def tokenize(self, root_path):
        with open(root_path, 'r') as f:
            text = f.read()
            self.chars = tuple(set(text))

        spacy_nlp = spacy.load('en_core_web_sm')
        spacy_nlp.max_length = 2 * len(text)

        x = spacy_nlp(text, disable=['parser', 'tagger', 'ner'])
        splitted = [token.text for token in x if
                    not (token.text.isspace() or (token.text[0].isdigit() and token.text[-1].isdigit()) or (token.text[0] in string.punctuation and token.text[-1] in string.punctuation))]
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
                                 aug_word_max=arr.size // 3, special_char=False, tokenizer = lambda x: x.split(), reverse_tokenizer = lambda x: x)
        augmented_data = augmentator.augment(" ".join(arr.ravel().tolist()))
        return np.array(augmented_data).reshape(arr.shape)

    def get_encodes(self, arr):
        arr = self.augment(arr)

        flat_arr = arr.ravel()
        splitted_encoded = np.array(list(map(lambda x: np.array([self.char2int[ch] for ch in x]), flat_arr)))

        first_char = list(map(lambda x: x[0], splitted_encoded))
        last_char = list(map(lambda x: x[-1], splitted_encoded))
        middle = splitted_encoded

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
        return np.shape(self.data)[0]

    def __getitem__(self, index):
        sequence = self.data[index]
        X = self.get_encodes(sequence.copy())
        y = np.vectorize(self.get_int)(sequence)
        return torch.from_numpy(X), torch.from_numpy(y)








class seq2seqDataset(data.Dataset):
    def __init__(self, root_path, seq_length, embed_dim, embed_vec_space):
        """Initialize the dataset"""

        self.root_path = root_path
        self.seq_length = seq_length
        self.tokenized_data = self.tokenize(root_path)
        self.embed = BPEmb(lang="en", vs=embed_vec_space, add_pad_emb=True)



    def tokenize(self, root_path):
  
        with open(root_path, 'r') as f:
            text = f.read()
        lt = len(text)
        print(lt)

        splitted = []
        start = 0

        while True:

          flag = (1 if self.seq_length >= lt - start else 0)
          cur_text = text[start: start + min(self.seq_length, lt - start)]
          last_chunk_len = (0 if cur_text[-1].isspace() else len(cur_text.split()[-1]))
          
          start += self.seq_length - last_chunk_len

          if last_chunk_len == len(cur_text.strip()):
            start += self.seq_length
            continue
          splitted.append(cur_text[:-last_chunk_len].strip())

          if flag:
            break

        return splitted

    def augment(self, seq):
        augmentator = nac.KeyboardAug(aug_char_min=0, aug_char_p=0.4, aug_word_p=0.5, aug_word_min=0,
                                 aug_word_max=len(seq) // 5, special_char=False)

        augmented_data = augmentator.augment(seq)
        return seq

    def get_encodes(self, seq, use_aug=False):
        if use_aug:
            seq = self.augment(seq)
        

        def padded_encode(x):
            res = np.full((self.seq_length, self.embed.dim), self.embed['<pad>'])
            enc = self.embed.encode_ids(x)
            
            res[:len(enc)] = self.embed.vectors[enc]
            length = len(enc)
            
            res = np.insert(res, 0, np.full((self.embed.dim), -1), 0)
            length += 1

            if use_aug:
              res = np.insert(res, length, np.full((self.embed.dim), -2), 0)
              length += 1

            return np.array([length]), res

        len_vec, res_arr = padded_encode(seq)  

                  
        
        # len_vec, perm_idx = torch.from_numpy(len_vec).sort(0, descending=True)
        # res_arr = res_arr[perm_idx]

        return len_vec, res_arr

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, index):
        sequence = self.tokenized_data[index]
        
        lengths_x, X = self.get_encodes(sequence, use_aug=True)
        lengths_y, Y = self.get_encodes(sequence)
        
        return tuple([torch.from_numpy(arr) for arr in (lengths_x, X, lengths_y, Y)])



# def get_encodes(arr, seq_length, use_aug=False):
#   if use_aug:
#     aug_rr = nac.KeyboardAug(aug_char_min=0, aug_char_max=None, aug_char_p=0.4, aug_word_p=0.4, aug_word_min=0, aug_word_max=arr.size//3, special_char=False)
    
#     augmented_data = list(map(lambda x: aug_rr.augment(x), arr.ravel()))
#     arr = np.array(augmented_data).reshape(arr.shape)
#   flat_arr = arr.ravel()

#   def padded_encode(x):
#     k = np.full((seq_length,), bpemb_en.vs)
#     enc = np.array(bpemb_en.encode_ids(x))
#     k[:enc.size] = enc


#     return enc.size, k

#   res_arr = np.empty((*flat_arr.shape, seq_length), dtype="int32")
#   len_vec = np.empty( (arr.shape[0]))
#   for i in range(flat_arr.size):

#     res = padded_encode(flat_arr[i])  
#     res_arr[i] = res[1]
#     len_vec[i] = res[0]

#   if not use_aug:
#     res_arr = np.insert(res_arr, 0, 1, 1)
#   len_vec, perm_idx = torch.from_numpy(len_vec).sort(0, descending=True)
#   res_arr = res_arr[perm_idx]

#   leng, res = len_vec, one_hot_encode(res_arr, bpemb_en.vs + 1)

#   leng += (1 if not use_aug else 0)
#   leng, res = leng, torch.from_numpy(res)

#   return leng, res

# def get_batches(arr, batch_size, seq_length):
#     '''Create a generator that returns batches of size
#        batch_size x seq_length from arr.
       
#        Arguments
#        ---------
#        arr: Array you want to make batches from
#        batch_size: Batch size, the number of sequences per batch
#        seq_length: Number of encoded chars in a sequence
#     '''
    
#     # total number of batches we can make
#     n_batches = len(arr)//batch_size
    
#     # Keep only enough characters to make full batches
#     arr = arr[:n_batches * batch_size]    

  

#     # Reshape into batch_size rows
#     arr = arr.reshape((batch_size, -1))
#     # iterate through the array, one sequence at a time
#     for n in range(0, arr.shape[1]):
#         # The features
#         base = arr[:, n:n+1]
#         # y = np.vectorize(get_int)(base)

#         x = base.copy()
#         # y = one_hot_encode(y, len(words))
#         lengths_x, x = get_encodes(x, seq_length, use_aug=True)
#         lengths_y, y = get_encodes(base, seq_length)
        
#         yield lengths_x, x, lengths_y, y

if __name__ == '__main__':
    ds = seq2seqDataset("data/big.txt", 40, 1000, 50)
