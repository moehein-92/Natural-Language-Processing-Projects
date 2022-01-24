import pandas as pd
import unicodedata
import re
from torch.utils.data import Dataset
import torch
import random
import os

from gensim.models import FastText
import numpy as np
import random
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm.notebook import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, corpus_bleu

# Converts the unicode file to ascii
def unicode_to_ascii(s):
    """Normalizes latin chars with accent to their canonical decomposition"""
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    '''
    Preprocess the sentence to add the start, end tokens and make them lower-case
    '''
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r'([?.!,¿])', r' \1 ', w)
    w = re.sub(r'[" "]+', ' ', w)

    w = re.sub(r'[^a-zA-Z?.!,¿]+', ' ', w)
    
    w = w.rstrip().strip()
    w = '<start> ' + w + ' <end>'
    return w


def pad_sequences(x, max_len):
    padded = np.zeros((max_len), dtype=np.int64)
    if len(x) > max_len:
        padded[:] = x[:max_len]
    else:
        padded[:len(x)] = x
    return padded


def preprocess_data_to_tensor(dataframe, src_vocab, trg_vocab):
    # Vectorize the input and target languages
    src_tensor = [[src_vocab.word2idx[s if s in src_vocab.vocab else '<unk>'] for s in es.split(' ')] for es in dataframe['es'].values.tolist()]
    trg_tensor = [[trg_vocab.word2idx[s if s in trg_vocab.vocab else '<unk>'] for s in eng.split(' ')] for eng in dataframe['eng'].values.tolist()]

    # Calculate the max_length of input and output tensor for padding
    max_length_src, max_length_trg = max(len(t) for t in src_tensor), max(len(t) for t in trg_tensor)
    print('max_length_src: {}, max_length_trg: {}'.format(max_length_src, max_length_trg))

    # Pad all the sentences in the dataset with the max_length
    src_tensor = [pad_sequences(x, max_length_src) for x in src_tensor]
    trg_tensor = [pad_sequences(x, max_length_trg) for x in trg_tensor]

    return src_tensor, trg_tensor, max_length_src, max_length_trg


def train_test_split(src_tensor, trg_tensor):
    '''
    Create training and test sets.
    '''
    total_num_examples = len(src_tensor) - int(0.2*len(src_tensor))
    src_tensor_train, src_tensor_test = src_tensor[:int(0.75*total_num_examples)], src_tensor[int(0.75*total_num_examples):total_num_examples]
    trg_tensor_train, trg_tensor_test = trg_tensor[:int(0.75*total_num_examples)], trg_tensor[int(0.75*total_num_examples):total_num_examples]

    return src_tensor_train, src_tensor_test, trg_tensor_train, trg_tensor_test

class Vocab_Lang():
    def __init__(self, vocab):
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}
        self.vocab = vocab
        
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 2 # +2 because of <pad> and <unk> token
            self.idx2word[index + 2] = word
        

class MyData(Dataset):
    def __init__(self, X, y):
        self.length = torch.LongTensor([np.sum(1 - np.equal(x, 0)) for x in X])
        self.data = torch.LongTensor(X)
        self.target = torch.LongTensor(y)
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y

    def __len__(self):
        return len(self.data)

def compute_FastText_embeddings(pd_dataframe, embedding_dim):
    """
    Given dataset (pd.DataFrame as used in the beginning), train FastText embeddings
    Return FastText trained model and embeddings vectors (np array [2 + vocab_size, embedding_dim])
    """
    
    print('Computing FastText Embeddings...')
    sentences = [sen.split() for sen in pd_dataframe]
    
    # Create FastText model to learn `embedding_dim` sized embedding vectors
    model = FastText(vector_size = embedding_dim)
    
    # Build vocab from sentences
    model.build_vocab(sentences=sentences)
    
    # Train model on sentences for 10 epochs
    model.train(sentences=sentences, epochs=10, total_examples=len(sentences))
    
    embedding_vec = model.wv.vectors
    
    # The sentences that we used to train the embedding don't contain '<pad>', or '<unk>' 
    # so add two all-zero or random rows in the beginning of the embedding np array for '<pad>' and '<unk>'
    newrow = np.zeros(256)
    embedding_vec = np.vstack([embedding_vec, newrow])
    embedding_vec = np.vstack([embedding_vec, newrow])

    return model, embedding_vec


class RnnEncoder(nn.Module):
    def __init__(self, pretrained_emb, vocab_size, embedding_dim, hidden_units):
        super(RnnEncoder, self).__init__()
        """
        Args:
            pretrained_emb: np.array, the pre-trained source embedding computed from compute_FastText_embeddings
            vocab_size: int, the size of the source vocabulary
            embedding_dim: the dimension of the embedding
            hidden_units: The number of features in the GRU hidden state
        """

        # Convert pretrained_emb from np.array to torch.FloatTensor
        self.pretrained_emb = torch.FloatTensor(pretrained_emb)

        # Initialize embedding layer with pretrained_emb
        self.embed = nn.Embedding.from_pretrained(self.pretrained_emb)
        
        # Initialize a single directional GRU with 1 layer and batch_first=False
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_units, num_layers=1, batch_first=False)


    def forward(self, x):
        """
        Args:
            X: source texts, [max_len, batch_size]

        Returns:
            output: [max_len, batch_size, hidden_units]
            hidden_state: [1, batch_size, hidden_units] 
        
        Pseudo-code:
        - Pass x through an embedding layer and pass the results through the recurrent net
        - Return output and hidden states from the recurrent net
        """

        embedded = self.embed(x)
        output, hidden_state = self.gru(embedded)
        
        return output, hidden_state


class RnnDecoder(nn.Module):
    def __init__(self, pretrained_emb, vocab_size, embedding_dim, hidden_units):
        super(RnnDecoder, self).__init__()
        
        # Convert pretrained_emb from np.array to torch.FloatTensor
        self.pretrained_emb = torch.FloatTensor(pretrained_emb)

        # Initialize embedding layer with pretrained_emb
        self.embed = nn.Embedding.from_pretrained(self.pretrained_emb)

        # Initialize layers to compute attention score
        self.W1 = nn.Linear(hidden_units, hidden_units)
        self.W2 = nn.Linear(hidden_units, hidden_units)
        self.va = nn.Linear(hidden_units, 1)

        # Initialize a single directional GRU with 1 layer and batch_first=True
        # NOTE: Input to your RNN will be the concatenation of your embedding vector and the context vector
        self.gru = nn.GRU(input_size=hidden_units + embedding_dim, hidden_size=hidden_units, num_layers=1, batch_first=True)

        # Initialize fully connected layer
        self.fc = nn.Linear(hidden_units, vocab_size)


    def compute_attention(self, dec_hs, enc_output):
       
        # Compute the attention scores for dec_hs & enc_output
        enc_output = enc_output.permute(1, 0, 2)
        dec_hs = dec_hs.premute(1, 0, 2)
        
        scores = self.va((nn.Tanh(self.W1(dec_hs) + self.W2(enc_output))))

        # Compute attention_weights by taking a softmax over your scores to normalize the distribution
        attention_weights = F.softmax(scores, dim=1)
        
        # Compute context_vector from attention_weights & enc_output
        context_vector = torch.sum((attention_weights * enc_output), dim = 2)
        
        # Return context_vector & attention_weights
        return context_vector, attention_weights
        

    def forward(self, x, dec_hs, enc_output):

       # Compute the context vector & attention weights by calling self.compute_attention(...) on the appropriate input
        context_vector, attention_weights = self.compute_attention(dec_hs, enc_output)

        # Obtain embedding vectors for your input x
        #     - Output size: [batch_size, 1, embedding_dim]
        embedded = self.embed(x)

        # Concatenate the context vector & the embedding vectors along the appropriate dimension
        input = torch.cat((context_vector.unsqueeze(1), embedded), dim=2) 

        # Feed this result through your RNN (along with the current hidden state) to get output and new hidden state
        #     - Output sizes: [batch_size, 1, hidden_units] & [1, batch_size, hidden_units] 
        output, dec_hs = self.gru(input, dec_hs)

        # Feed the output of your RNN through linear layer to get (unnormalized) output distribution (don't call softmax!)
        fc_out = self.fc(output)
 
        return fc_out, dec_hs, attention_weights

def create_positional_embedding(max_len, embed_dim):

    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
    pe = torch.zeros(max_len, 1, embed_dim)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)

    return pe
    

class TransformerEncoder(nn.Module):
    def __init__(self, pretrained_emb, src_vocab_size, embedding_dim, num_heads,
        num_layers, dim_feedforward, max_len_src, device):
        super(TransformerEncoder, self).__init__()
        self.device = device

        # Create positional embedding matrix
        self.position_embedding = create_positional_embedding(max_len_src, embedding_dim).to(device)
        self.register_buffer('positional_embedding', self.position_embedding) # this informs the model that position_embedding is not a learnable parameter

        # Convert pretrained_emb from np.array to torch.FloatTensor
        self.pretrained_emb = torch.FloatTensor(pretrained_emb)

        # Initialize embedding layer with pretrained_emb
        self.embed = nn.Embedding.from_pretrained(self.pretrained_emb)

        # Dropout layer
        self.dropout = nn.Dropout()

        # Initialize a nn.TransformerEncoder model (you'll need to use embedding_dim,
        # num_layers, num_heads, & dim_feedforward here)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == 0 # padding idx
        return src_mask.to(self.device) # (batch_size, max_src_len)

    def forward(self, x):

        # Pass x through the word embedding
        embedded = self.embed(x)

        # Add positional embedding to the word embedding, then apply dropout
        embedded = embedded + self.position_embedding[:embedded.size(0)]
        drop = self.dropout(embedded)

        # Call make_src_mask(x) to compute a mask: this tells us which indexes in x
        # are padding, which we want to ignore for the self-attention
        src_mask = self.make_src_mask(x)

        # Call the encoder, with src_key_padding_mask = src_mask
        output = self.transformer_encoder(drop, src_key_padding_mask=src_mask)

        return output    


class TransformerDecoder(nn.Module):
    def __init__(self, pretrained_emb, trg_vocab_size, embedding_dim, num_heads,
        num_layers, dim_feedforward, max_len_trg, device):
        super(TransformerDecoder, self).__init__()
        self.device = device

        # Create positional embedding matrix
        self.position_embedding = create_positional_embedding(max_len_trg, embedding_dim).to(device)
        self.register_buffer('positional_embedding', self.position_embedding) # this informs the model that positional_embedding is not a learnable parameter

        # Convert pretrained_emb from np.array to torch.FloatTensor
        self.pretrained_emb = torch.FloatTensor(pretrained_emb)

        # Initialize embedding layer with pretrained_emb
        self.embed = nn.Embedding.from_pretrained(self.pretrained_emb)

        # Dropout layer
        self.dropout = nn.Dropout()

        # Initialize a nn.TransformerDecoder model (you'll need to use embedding_dim,
        # num_layers, num_heads, & dim_feedforward here)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final fully connected layer
        self.fc = nn.Linear(embedding_dim, trg_vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, dec_in, enc_out):

        # Compute input word and positional embeddings in similar manner to encoder
        embedded = self.embed(dec_in)
        embedded = embedded + self.position_embedding[:embedded.size(0)]
        drop = self.dropout(embedded)

        # Call generate_square_subsequent_mask() to compute a mask: this time,
        # the mask is to prevent the decoder from attending to tokens in the "future".
        # In other words, at time step i, the decoder should only attend to tokens
        # 1 to i-1.
        trg_mask = self.generate_square_subsequent_mask(dec_in)

        # Call the decoder, with trg_mask = trg_mask
        output = self.transformer_decoder(tgt=drop, memory=enc_out,tgt_mask=trg_mask)

        # Run the output through the fully-connected layer and return it
        output = self.fc(output)

        return output    




if __name__ == '__main__':
    lines = open('spa.txt', encoding='UTF-8').read().strip().split('\n')
    total_num_examples = 50000 
    original_word_pairs = [[w for w in l.split('\t')][:2] for l in lines[:total_num_examples]]
    random.seed(42)
    random.shuffle(original_word_pairs)
    dat = pd.DataFrame(original_word_pairs, columns=['eng', 'es'])
    print(dat) # Visualize the data

    data = dat.copy()
    data['eng'] = dat.eng.apply(lambda w: preprocess_sentence(w))
    data['es'] = dat.es.apply(lambda w: preprocess_sentence(w))
    print(data) # visualizing the data

    # HYPERPARAMETERS (You may change these if you want, though you shouldn't need to)
    BATCH_SIZE = 64
    EMBEDDING_DIM = 256

    fasttext_model_src, embedding_src = compute_FastText_embeddings(data['es'], EMBEDDING_DIM)
    fasttext_model_trg, embedding_trg = compute_FastText_embeddings(data['eng'], EMBEDDING_DIM)

    src_vocab = Vocab_Lang(fasttext_model_src.wv.vocab)
    trg_vocab = Vocab_Lang(fasttext_model_trg.wv.vocab)
    src_tensor, trg_tensor, max_length_src, max_length_trg = preprocess_data_to_tensor(data, src_vocab, trg_vocab)
    src_tensor_train, src_tensor_val, trg_tensor_train, trg_tensor_val = train_test_split(src_tensor, trg_tensor)

    # create train and val datasets
    train_dataset = MyData(src_tensor_train, trg_tensor_train)
    train_dataset = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)

    test_dataset = MyData(src_tensor_val, trg_tensor_val)
    test_dataset = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=False)

    idxes = random.choices(range(len(train_dataset.dataset)), k=5)
    src, trg =  train_dataset.dataset[idxes]
    print('Source:', src)
    print('Target:', trg)