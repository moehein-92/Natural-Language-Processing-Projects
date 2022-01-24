from collections import defaultdict
import numpy as np
import torch
from torch.functional import einsum
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import torchtext 
import random
from tqdm.notebook import tqdm

device = torch.device('cuda') #if torch.cuda.is_available() else 'cpu')

if __name__=='__main__':
    print('Using device:', device)

def preprocess(review):
    '''
    Simple preprocessing function.
    '''
    res = []
    for x in review.split(' '):
        remove_beg=True if x[0] in {'(', '"', "'"} else False
        remove_end=True if x[-1] in {'.', ',', ';', ':', '?', '!', '"', "'", ')'} else False
        if remove_beg and remove_end: res += [x[0], x[1:-1], x[-1]]
        elif remove_beg: res += [x[0], x[1:]]
        elif remove_end: res += [x[:-1], x[-1]]
        else: res += [x]
    return res


train_data = torchtext.datasets.IMDB(root='.data', split='train')
train_data = list(train_data)
train_data = [(x[0], preprocess(x[1])) for x in train_data]
train_data, test_data = train_data[0:10000] + train_data[12500:12500+10000], train_data[10000:12500] + train_data[12500+10000:], 


PAD = '<PAD>'
END = '<END>'
UNK = '<UNK>'


class TextDataset(data.Dataset):
    def __init__(self, examples, split, threshold, max_len, idx2word=None, word2idx=None):

        self.examples = examples
        assert split in {'train', 'val', 'test'}
        self.split = split
        self.threshold = threshold
        self.max_len = max_len

        # Dictionaries
        self.idx2word = idx2word
        self.word2idx = word2idx
        if split == 'train':
            self.build_dictionary()
        self.vocab_size = len(self.word2idx)
        
        # Convert text to indices
        self.textual_ids = []
        self.convert_text()

    def build_dictionary(self): 
        '''
        Build the dictionaries idx2word and word2idx. This is only called when split='train', as these
        dictionaries are passed in to the __init__(...) function otherwise. Be sure to use self.threshold
        to control which words are assigned indices in the dictionaries.
        Returns nothing.
        '''
        assert self.split == 'train'
        
        self.idx2word = {0:PAD, 1:END, 2: UNK}
        self.word2idx = {PAD:0, END:1, UNK: 2}

        # Count the frequencies of all words in the training data (self.examples)
        # Assign idx (starting from 3) to all words having word_freq >= self.threshold
        # Make sure you call word.lower() on each word to convert it to lowercase
        
        self.freq = defaultdict(int)

        for example in self.examples:
            word_list = example[1]
            for word in word_list:
                word = word.lower()
                self.freq[word] += 1

        idx = 3
        for word in self.freq:
            if self.freq[word] >= self.threshold:
                self.idx2word[idx] = word
                self.word2idx[word] = idx
                idx += 1


    def convert_text(self):
        '''
        Convert each review in the dataset (self.examples) to a list of indices, given by self.word2idx.
        Store this in self.textual_ids; returns nothing.
        '''

        # replace a word with the <UNK> token if it does not exist in the word2idx dictionary.
        # append the <END> token to the end of each review.

        self.labels = []

        for example in self.examples:
            label, word_list = example
            lst = []
            for word in word_list:
                if word in self.word2idx:
                    lst.append(self.word2idx[word])
                else:
                    lst.append(self.word2idx[UNK])
            lst.append(self.word2idx[END])
            self.textual_ids.append(lst)
            self.labels.append(label)

    def get_text(self, idx):
        '''
        Return the review at idx as a long tensor (torch.LongTensor) of integers corresponding to the words in the review.
        You may need to pad as necessary (see above).
        '''
    
        review = self.textual_ids[idx]

        if len(review) >= self.max_len:
            return torch.LongTensor(review[:self.max_len])

        elif len(review) < self.max_len:
            while len(review) < self.max_len:
                review.append(self.word2idx[PAD])
            return torch.LongTensor(review)


    def get_label(self, idx):
        '''
        This function should return the value 1 if the label for idx in the dataset is 'positive', 
        and 0 if it is 'negative'. The return type should be torch.LongTensor.
        '''
        
        label = self.labels[idx]
        if label == 'pos':
            tag = 1
        elif label == 'neg':
            tag = 0

        return torch.LongTensor([tag]).squeeze()

    def __len__(self):
        '''
        Return the number of reviews (int value) in the dataset
        '''
        return len(self.examples)


    def __getitem__(self, idx):
        '''
        Return the review, and label of the review specified by idx.
        '''
        
        label = self.get_label(idx)
        text = self.get_text(idx)
        
        return text, label


class CNN(nn.Module):
    def __init__(self, vocab_size, embed_size, out_channels, filter_heights, stride, dropout, num_classes, pad_idx):
        super(CNN, self).__init__()
        
        # Embedding Layer
        self.embed = nn.Embedding(vocab_size, embed_size, pad_idx)
        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(filter_heights[0], embed_size), stride=stride)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(filter_heights[1], embed_size), stride=stride)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(filter_heights[2], embed_size), stride=stride)
        # Dropout Layer
        self.dropout = nn.Dropout(p=dropout, inplace=False)
        # Linear Layer
        self.fc1 = nn.Linear(out_channels * len(filter_heights), num_classes)

    def forward(self, texts):

        # Add word embeddings
        e = self.embed(texts)

        # Add an extra dimension for CNN
        e = e.unsqueeze(1)

        # Apply CNN
        c1 = F.relu(self.conv1(e)).squeeze(3)
        c2 = F.relu(self.conv2(e)).squeeze(3)
        c3 = F.relu(self.conv3(e)).squeeze(3)
        # Max pooling
        p1 = F.max_pool1d(c1, c1.shape[2]).squeeze(2)        
        p2 = F.max_pool1d(c2, c2.shape[2]).squeeze(2)
        p3 = F.max_pool1d(c3, c3.shape[2]).squeeze(2)

        cat = torch.cat((p1, p2, p3), dim=1)
        d = self.dropout(cat)

        final = self.fc1(d)

        return final



def train_model(model, num_epochs, data_loader, optimizer, criterion):
    print('Training Model...')
    model.train()
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        epoch_acc = 0
        for texts, labels in data_loader:
            texts = texts.to(device) # shape: [batch_size, MAX_LEN]
            labels = labels.to(device) # shape: [batch_size]

            optimizer.zero_grad()

            output = model(texts)
            acc = accuracy(output, labels)
            
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        print('[TRAIN]\t Epoch: {:2d}\t Loss: {:.4f}\t Train Accuracy: {:.2f}%'.format(epoch+1, epoch_loss/len(data_loader), 100*epoch_acc/len(data_loader)))
    print('Model Trained!\n')

def count_parameters(model):
    """
    Count number of trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def accuracy(output, labels):
    """
    Returns accuracy per batch
    output: Tensor [batch_size, n_classes]
    labels: LongTensor [batch_size]
    """
    preds = output.argmax(dim=1) # find predicted class
    correct = (preds == labels).sum().float() # convert into float for division 
    acc = correct / len(labels)
    return acc


def evaluate(model, data_loader, criterion):
    print('Evaluating performance on the test dataset...')
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    all_predictions = []
    print("\nSOME PREDICTIONS FROM THE MODEL:")
    for texts, labels in tqdm(data_loader):
        texts = texts.to(device)
        labels = labels.to(device)
        
        output = model(texts)
        acc = accuracy(output, labels)
        pred = output.argmax(dim=1)
        all_predictions.append(pred)
        
        loss = criterion(output, labels)
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()

        if random.random() < 0.0015:
            print("Input: "+' '.join([data_loader.dataset.idx2word[idx] for idx in texts[0].tolist() if idx not in {data_loader.dataset.word2idx[PAD], data_loader.dataset.word2idx[END]}]))
            print("Prediction:", pred.item(), '\tCorrect Output:', labels.item(), '\n')

    full_acc = 100*epoch_acc/len(data_loader)
    full_loss = epoch_loss/len(data_loader)
    print('[TEST]\t Loss: {:.4f}\t Accuracy: {:.2f}%'.format(full_loss, full_acc))
    predictions = torch.cat(all_predictions)
    return predictions, full_acc, full_loss


if __name__=='__main__':
    THRESHOLD = 5 # Don't change this
    MAX_LEN = 100 # Don't change this
    BATCH_SIZE = 32 # Feel free to try other batch sizes

    train_Ds = TextDataset(train_data, 'train', THRESHOLD, MAX_LEN)
    train_loader = torch.utils.data.DataLoader(train_Ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)

    test_Ds = TextDataset(test_data, 'test', THRESHOLD, MAX_LEN, train_Ds.idx2word, train_Ds.word2idx)
    test_loader = torch.utils.data.DataLoader(test_Ds, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    cnn_model = CNN(vocab_size = train_Ds.vocab_size, # Don't change this
                embed_size = 128, 
                out_channels = 64, 
                filter_heights = [2, 3, 4], 
                stride = 1, 
                dropout = 0.5, 
                num_classes = 2, # Don't change this
                pad_idx = train_Ds.word2idx[PAD]) # Don't change this

    # Put your model on the device (cuda or cpu)
    cnn_model = cnn_model.to(device)
    
    print('The model has {:,d} trainable parameters'.format(count_parameters(cnn_model)))

    LEARNING_RATE = 5e-4 # Feel free to try other learning rates

    # Define the loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Define the optimizer
    optimizer = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)

    N_EPOCHS = 10 # Feel free to change this
    
    # train model for N_EPOCHS epochs
    train_model(cnn_model, N_EPOCHS, train_loader, optimizer, criterion)

    # Compute test data accuracy
    evaluate(cnn_model, test_loader, criterion) 
    
    # Save model
    torch.save(cnn_model, "cnn.pt")