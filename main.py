


#---------------Chatbot using only using encoder and decoder----------------------
import os
# import yaml
#
# dir_path = 'raw_data'
# # files_list = os.listdir(dir_path + os.sep)
#
# questions = list()
# answers = list()
# unique_tokens = set()
#
# stream = open( 'sample.yaml' , 'rb')
# docs = yaml.safe_load(stream)
# conversations = docs['conversations']
# for con in conversations:
#         if len( con ) > 2 :
#             questions.append(con[0])
#             replies = con[ 1 : ]
#             ans = ''
#             for rep in replies:
#                 ans += ' ' + rep
#             answers.append( ans )
#         elif len( con )> 1:
#             questions.append(con[0])
#             answers.append(con[1])
# # for filepath in files_list:
# answers_with_tags = list()
# for i in range(len(answers)):
#     if type(answers[i]) == str:
#         answers_with_tags.append(answers[i])
#     else:
#         questions.pop(i)
#
# answers = list()
# for i in range(len(answers_with_tags)):
#     answers.append('<START> ' + answers_with_tags[i] + ' <END>')
#     print(answers)
#
# from keras_preprocessing import text
#
# tokenizer = text.Tokenizer()
# tokenizer.fit_on_texts(questions + answers)
# VOCAB_SIZE = len(tokenizer.word_index) + 1
# import re
# def tokenize(sentences):
#     tokens_list = []
#     vocabulary = []
#     for sentence in sentences:
#         sentence = sentence.lower()
#         sentence = re.sub('[^a-zA-Z]', ' ', sentence)
#         tokens = sentence.split()
#         vocabulary += tokens
#         tokens_list.append(tokens)
#     return tokens_list, vocabulary
# from keras import layers, activations, models, preprocessing
# import numpy as np
# import keras
# from keras import preprocessing, utils
# tokenized_questions = tokenizer.texts_to_sequences(questions)
# maxlen_questions = max(len(x) for x in tokenized_questions)
# padded_questions = preprocessing.sequence.pad_sequences(tokenized_questions, maxlen=maxlen_questions, padding='post')
# encoder_input_data = np.array(padded_questions)
# tokenized_answers = tokenizer.texts_to_sequences(answers)
# maxlen_answers = max(len(x) for x in tokenized_answers)
# padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding='post')
# decoder_input_data = np.array(padded_answers)
# tokenized_answers = tokenizer.texts_to_sequences(answers)
# for i in range(len(tokenized_answers)):
#     tokenized_answers[i] = tokenized_answers[i][1:]
#
# padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding='post')
# onehot_answers = utils.to_categorical(padded_answers, VOCAB_SIZE)
# decoder_output_data = np.array(onehot_answers)
#
# encoder_inputs = keras.layers.Input(shape=(maxlen_questions,))
# encoder_embedding = keras.layers.Embedding(VOCAB_SIZE, 200, mask_zero=True)(encoder_inputs)
# encoder_outputs, state_h, state_c = keras.layers.LSTM(200, return_state=True)(encoder_embedding)
# encoder_states = [state_h, state_c]
#
# decoder_inputs = keras.layers.Input(shape=(maxlen_answers,))
# decoder_embedding = keras.layers.Embedding(VOCAB_SIZE, 200, mask_zero=True)(decoder_inputs)
# decoder_lstm = keras.layers.LSTM(200, return_state=True, return_sequences=True)
# decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
# decoder_dense = keras.layers.Dense(VOCAB_SIZE, activation=keras.activations.softmax)
# output = decoder_dense(decoder_outputs)
# model = keras.models.Model([encoder_inputs, decoder_inputs], output)
# model.compile(optimizer=keras.optimizers.RMSprop(), loss='categorical_crossentropy')
#
# model.summary()
# model.fit([encoder_input_data , decoder_input_data], decoder_output_data, batch_size=50, epochs=150 )
# model.save( 'model.h5' )
# def make_inference_models():
#     encoder_model = keras.models.Model(encoder_inputs, encoder_states)
#     decoder_state_input_h = keras.layers.Input(shape=(200,))
#     decoder_state_input_c = keras.layers.Input(shape=(200,))
#     decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
#     decoder_outputs, state_h, state_c = decoder_lstm(
#         decoder_embedding, initial_state=decoder_states_inputs)
#     decoder_states = [state_h, state_c]
#     decoder_outputs = decoder_dense(decoder_outputs)
#     decoder_model = keras.models.Model(
#         [decoder_inputs] + decoder_states_inputs,
#         [decoder_outputs] + decoder_states)
#     return encoder_model, decoder_model
#
# enc_model, dec_model = make_inference_models()
# def str_to_tokens(sentence : str ):
#     words = sentence.lower().split()
#     tokens_list = []
#     for word in words:
#         try:
#             tokens_list.append(tokenizer.word_index[word])
#         except KeyError:
#             tokens_list.append(tokenizer.word_index['<OOV>'])
#     return preprocessing.sequence.pad_sequences([tokens_list], maxlen=maxlen_questions, padding='post')
#
# import tensorflow as tf
# import pickle
# for _ in range(10):
#     states_values = enc_model.predict(str_to_tokens(input('Enter question : ')))
#     empty_target_seq = np.zeros((1, 1))
#     empty_target_seq[0, 0] = tokenizer.word_index['start']
#     stop_condition = False
#     decoded_translation = ''
#     while not stop_condition:
#         dec_outputs, h, c = dec_model.predict([empty_target_seq] + states_values)
#         sampled_word_index = np.argmax(dec_outputs[0, -1, :])
#         sampled_word = None
#         for word, index in tokenizer.word_index.items():
#             if sampled_word_index == index:
#                 decoded_translation += ' {}'.format(word)
#                 sampled_word = word
#         if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
#             stop_condition = True
#         empty_target_seq = np.zeros((1, 1))
#         empty_target_seq[0, 0] = sampled_word_index
#         states_values = [h, c]
#     print(decoded_translation)


#----------------------------xxxxxxxxxxxxxxxxxxxxxxxxxxx-------------------------------


from collections import Counter
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.utils.data
import math
import torch.nn.functional as F
max_len = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
questions=[]
answers=[]
l=[]
with open('dummy.txt', 'r') as file:
    data = file.readlines()
    for i in range(0, len(data), 2):  # Iterate over every other line
        question = data[i].strip()  # Get question
        answer = data[i + 1].strip()  # Get answer
        questions.append(question)
        answers.append(answer)
        l.append([question,answer])
questions_lower = [question.lower() for question in questions]
answers_lower = [answer.lower() for answer in answers]
print(l)
data = l
# Splitting each sentence into words and nesting them
nested_data = [[[word for word in sentence.split()] for sentence in inner_list] for inner_list in data]
with open('dummy.txt', 'r') as k:
    data = k.readlines()

# def remove_punc(string):
#     punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
#     no_punct = ""
#     for char in string:
#         if char not in punctuations:
#             no_punct = no_punct + char  # space is also a character
#     return no_punct.lower()
#
#
pairs = []
pairs=nested_data
pairs2=[]
for i in range(len(data)):
    qa_pairs = []
    if i == len(data) - 1:
        break
    first = data[i];
    second = data[i + 1]
    qa_pairs.append(first.split()[:max_len])
    qa_pairs.append(second.split()[:max_len])
    pairs2.append(qa_pairs)

word_freq = Counter()
for pair in pairs2:
    word_freq.update(pair[0])
    word_freq.update(pair[1])

min_word_freq =0
words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
# print(words)
word_map = {k: v + 1 for v, k in enumerate(words)}
word_map['<unk>'] = len(word_map) + 1
word_map['<start>'] = len(word_map) + 1
word_map['<end>'] = len(word_map) + 1
word_map['<pad>'] = 0
# print(word_map)
with open('WORDMAP_corpus.json', 'w') as j:
    json.dump(word_map, j)
def encode_question(words, word_map):
    enc_c = [word_map.get(word, word_map['<unk>']) for word in words] + [word_map['<pad>']] * (max_len - len(words))
    return enc_c

def encode_reply(words, word_map):
    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in words] + \
    [word_map['<end>']] + [word_map['<pad>']] * (max_len - len(words))
    return enc_c
#
# #
pairs_encoded = []
for pair in pairs:
    qus = encode_question(pair[0], word_map)
    # print(pair[0])
    ans = encode_reply(pair[1], word_map)
    # print(pair[1])

    pairs_encoded.append([qus, ans])
with open('pairs_encoded.json', 'w') as p:
    json.dump(pairs_encoded, p)

class Dataset(Dataset):
    def __init__(self):
        self.pairs = json.load(open('pairs_encoded.json'))
        self.dataset_size = len(self.pairs)

    def __getitem__(self, i):
        question = torch.LongTensor(self.pairs[i][0])
        reply = torch.LongTensor(self.pairs[i][1])
        return question, reply

    def __len__(self):
        return self.dataset_size

train_loader = torch.utils.data.DataLoader(Dataset(),
                                           batch_size = 30,
                                           shuffle=True,
                                           pin_memory=True)


def create_masks(question, reply_input, reply_target):
    def subsequent_mask(size):
        mask = torch.triu(torch.ones(size, size)).transpose(0, 1).type(dtype=torch.uint8)
        return mask.unsqueeze(0)
    question_mask = question != 0
    question_mask = question_mask.to(device)
    question_mask = question_mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, max_words)
    reply_input_mask = reply_input != 0
    reply_input_mask = reply_input_mask.unsqueeze(1)  # (batch_size, 1, max_words)
    reply_input_mask = reply_input_mask & subsequent_mask(reply_input.size(-1)).type_as(reply_input_mask.data)
    reply_input_mask = reply_input_mask.unsqueeze(1)  # (batch_size, 1, max_words, max_words)
    reply_target_mask = reply_target != 0  # (batch_size, max_words)
    return question_mask, reply_input_mask, reply_target_mask


class Embeddings(nn.Module):
    """
    Implements embeddings of the words and adds their positional encodings.
    """

    def __init__(self, vocab_size, d_model, max_len=50, num_layers=6):
        super(Embeddings, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(0.1)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = self.create_positinal_encoding(max_len, self.d_model)  # (1, max_len, d_model)
        self.te = self.create_positinal_encoding(num_layers, self.d_model)  # (1, num_layers, d_model)
        self.dropout = nn.Dropout(0.1)

    def create_positinal_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model).to(device)
        for pos in range(max_len):  # for each position of the word
            for i in range(0, d_model, 2):  # for each dimension of the each position
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)  # include the batch size
        return pe
    def forward(self, embedding, layer_idx):
        if layer_idx == 0:
            embedding = self.embed(embedding) * math.sqrt(self.d_model)
        embedding += self.pe[:,
                     :embedding.size(1)]  # pe will automatically be expanded with the same batch size as encoded_words
        # embedding: (batch_size, max_len, d_model), te: (batch_size, 1, d_model)
        embedding += self.te[:, layer_idx, :].unsqueeze(1).repeat(1, embedding.size(1), 1)
        embedding = self.dropout(embedding)
        return embedding


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model):
        super(MultiHeadAttention, self).__init__()
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.dropout = nn.Dropout(0.1)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.concat = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask):
        """
        query, key, value of shape: (batch_size, max_len, 512)
        mask of shape: (batch_size, 1, 1, max_words)
        """
        # (batch_size, max_len, 512)
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        # (batch_size, max_len, 512) --> (batch_size, max_len, h, d_k) --> (batch_size, h, max_len, d_k)
        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        # (batch_size, h, max_len, d_k) matmul (batch_size, h, d_k, max_len) --> (batch_size, h, max_len, max_len)
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(query.size(-1))
        scores = scores.masked_fill(mask == 0, -1e9)  # (batch_size, h, max_len, max_len)
        weights = F.softmax(scores, dim=-1)  # (batch_size, h, max_len, max_len)
        weights = self.dropout(weights)
        # (batch_size, h, max_len, max_len) matmul (batch_size, h, max_len, d_k) --> (batch_size, h, max_len, d_k)
        context = torch.matmul(weights, value)
        context = context.permute(0,2,1,3).contiguous().view(context.shape[0], -1, self.heads * self.d_k)
        # (batch_size, max_len, h * d_k)
        interacted = self.concat(context)
        return interacted

class FeedForward(nn.Module):
    def __init__(self, d_model, middle_dim=2048):
            super(FeedForward, self).__init__()
            self.fc1 = nn.Linear(d_model, middle_dim)
            self.fc2 = nn.Linear(middle_dim, d_model)
            self.dropout = nn.Dropout(0.1)

    def forward(self, x):
            out = F.relu(self.fc1(x))
            out = self.fc2(self.dropout(out))
            return out

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(EncoderLayer, self).__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, embeddings, mask):
        interacted = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, mask))
        interacted = self.layernorm(interacted + embeddings)
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        encoded = self.layernorm(feed_forward_out + interacted)
        return encoded


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(DecoderLayer, self).__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadAttention(heads, d_model)
        self.src_multihead = MultiHeadAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model)
        self.dropout = nn.Dropout(0.1)
    def forward(self, embeddings, encoded, src_mask, target_mask):
        query = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, target_mask))
        query = self.layernorm(query + embeddings)
        interacted = self.dropout(self.src_multihead(query, encoded, encoded, src_mask))
        interacted = self.layernorm(interacted + query)
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        decoded = self.layernorm(feed_forward_out + interacted)
        return decoded


class Transformer(nn.Module):
    def __init__(self, d_model, heads, num_layers, word_map):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.vocab_size = len(word_map)
        self.embed = Embeddings(self.vocab_size, d_model, num_layers=num_layers)
        self.encoder = EncoderLayer(d_model, heads)
        self.decoder = DecoderLayer(d_model, heads)
        self.logit = nn.Linear(d_model, self.vocab_size)

    def encode(self, src_embeddings, src_mask):
        for i in range(self.num_layers):
            src_embeddings = self.embed(src_embeddings, i)
            src_embeddings = self.encoder(src_embeddings, src_mask)
        return src_embeddings

    def decode(self, tgt_embeddings, target_mask, src_embeddings, src_mask):
        for i in range(self.num_layers):
            tgt_embeddings = self.embed(tgt_embeddings, i)
            # print("embeddings")
            # print(tgt_embeddings)
            tgt_embeddings = self.decoder(tgt_embeddings, src_embeddings, src_mask, target_mask)
        return tgt_embeddings

    def forward(self, src_words, src_mask, target_words, target_mask):
        encoded = self.encode(src_words, src_mask)
        decoded = self.decode(target_words, target_mask, encoded, src_mask)
        out = F.log_softmax(self.logit(decoded), dim=2)
        return out

class AdamWarmup:
    def __init__(self, model_size, warmup_steps, optimizer):
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.current_step = 0
        self.lr = 0

    def get_lr(self):
        return self.model_size ** (-0.5) * min(self.current_step ** (-0.5),
                                               self.current_step * self.warmup_steps ** (-1.5))
    def step(self):
        # Increment the number of steps each time we call the step function
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        # update the learning rate
        self.lr = lr
        self.optimizer.step()
class LossWithLS(nn.Module):
    def __init__(self, size, smooth):
        super(LossWithLS, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='none')  # Use reduction='none' instead of size_average=False, reduce=False
        self.confidence = 1.0 - smooth
        self.smooth = smooth
        self.size = size

    def forward(self, prediction, target, mask):
        """
        prediction of shape: (batch_size, max_words, vocab_size)
        target and mask of shape: (batch_size, max_words)
        """
        prediction = prediction.view(-1, prediction.size(-1))  # (batch_size * max_words, vocab_size)
        target = target.contiguous().view(-1)  # (batch_size * max_words)
        mask = mask.float().view(-1)

        labels = torch.full_like(prediction, self.smooth / (self.size - 1))
        labels.scatter_(1, target.unsqueeze(1), self.confidence)

        loss = self.criterion(prediction, labels)  # (batch_size * max_words, vocab_size)
        loss = (loss.sum(1) * mask).sum() / mask.sum()

        return loss
#--------------------------training part-----------------------------------------------------------
d_model = 512
heads = 8
num_layers =1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 200
print(device)
with open('WORDMAP_corpus.json', 'r') as j:
        word_map = json.load(j)
transformer = Transformer(d_model=d_model, heads=heads, num_layers=num_layers, word_map=word_map)
transformer = transformer.to(device)
adam_optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
transformer_optimizer = AdamWarmup(model_size=d_model, warmup_steps=4000, optimizer=adam_optimizer)
criterion = LossWithLS(len(word_map), 0.2)

def train(train_loader, transformer, criterion, epoch):
    transformer.train()
    sum_loss = 0
    count = 0
    for i, (question, reply) in enumerate(train_loader):
        samples = question.shape[0]
        # Move to device
        question = question.to(device)
        reply = reply.to(device)
        # Prepare Target Data
        reply_input = reply[:, :-1]
        reply_target = reply[:, 1:]
        question_mask, reply_input_mask, reply_target_mask = create_masks(question, reply_input, reply_target)
        # Get the transformer outputs
        out = transformer(question, question_mask, reply_input, reply_input_mask)
        # Compute the loss
        loss = criterion(out, reply_target, reply_target_mask)
        # Backprop
        transformer_optimizer.optimizer.zero_grad()
        loss.backward()
        transformer_optimizer.step()
        sum_loss += loss.item() * samples
        count += samples
        if i % 100 == 0:
            print("Epoch [{}][{}/{}]\tLoss: {:.3f}".format(epoch, i, len(train_loader), sum_loss / count))

def evaluate(transformer, question, question_mask, max_len, word_map):
    """
    Performs Greedy Decoding with a batch size of 1
    """
    rev_word_map = {v: k for k, v in word_map.items()}
    transformer.eval()
    print("Is in evaluation mode:", not transformer.training)
    start_token = word_map['<start>']
    encoded = transformer.encode(question, question_mask)
    words = torch.LongTensor([[start_token]]).to(device)

    for step in range(max_len - 1):
        size = words.shape[1]
        target_mask = torch.triu(torch.ones(size, size)).transpose(0, 1).type(dtype=torch.uint8)
        target_mask = target_mask.to(device).unsqueeze(0).unsqueeze(0)
        decoded = transformer.decode(words, target_mask, encoded, question_mask)
        predictions = transformer.logit(decoded[:, -1])
        _, next_word = torch.max(predictions, dim=1)
        next_word = next_word.item()
        if next_word == word_map['<end>']:
            break
        words = torch.cat([words, torch.LongTensor([[next_word]]).to(device)], dim=1)  # (1,step+2)
        # Construct Sentence
    if words.dim() == 2:
            words = words.squeeze(0)
            words = words.tolist()

    sen_idx = [w for w in words if w not in {word_map['<start>']}]
    sentence = ' '.join([rev_word_map[sen_idx[k]] for k in range(len(sen_idx))])

    return sentence

for epoch in range(epochs):
    train(train_loader, transformer, criterion, epoch)

    # state = {'epoch': epoch, 'transformer': transformer, 'transformer_optimizer': transformer_optimizer}
    # torch.save(state, 'checkpoint_' + str(epoch) + '.pth.tar')
    if epoch == epochs - 1:
        state = {'epoch': epoch, 'transformer': transformer, 'transformer_optimizer': transformer_optimizer}
        torch.save(state, 'checkpoint_last_epoch.pth.tar')

# checkpoint = torch.load('checkpoint_143.pth.tar')
checkpoint = torch.load('checkpoint_last_epoch.pth.tar')
transformer = checkpoint['transformer']


while(1):
    question = input("Question: ")
    if question == 'quit':
        break
    max_len = input("Maximum Reply Length: ")
    enc_qus = [word_map.get(word, word_map['<unk>']) for word in question.split()]
    question = torch.LongTensor(enc_qus).to(device).unsqueeze(0)
    question_mask = (question!=0).to(device).unsqueeze(1).unsqueeze(1)
    sentence = evaluate(transformer, question, question_mask, int(max_len), word_map)
    print(sentence)