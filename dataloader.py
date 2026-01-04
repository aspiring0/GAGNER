import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import prettytable as pt
from tqdm import tqdm
from transformers import AutoTokenizer
import os
import utils
from data_test import seq_gaz
from uitl.gazetter import Gazetteer
from uitl.alphabet import Alphabet

os.environ["TOKENIZERS_PARALLELISM"] = "false"

dis2idx = np.zeros((1000), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


class Vocabulary(object):
    PAD = '<pad>'
    UNK = '<unk>'
    SUC = '<suc>'

    def __init__(self):
        self.label2id = {self.PAD: 0, self.SUC: 1}
        self.id2label = {0: self.PAD, 1: self.SUC}

    def add_label(self, label):
        label = label.lower()
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)
            self.id2label[self.label2id[label]] = label

        assert label == self.id2label[self.label2id[label]]

    def __len__(self):
        return len(self.label2id)

    def label_to_id(self, label):
        label = label.lower()
        return self.label2id[label]

    def id_to_label(self, i):
        return self.id2label[i]


def collate_fn(data):
    bert_inputs,  grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text = map(list,zip(*data))

    max_tok = np.max(sent_length)
    sent_length = torch.LongTensor(sent_length)
    max_pie = np.max([x.shape[0] for x in bert_inputs])
    bert_inputs = pad_sequence(bert_inputs, True)
    batch_size = bert_inputs.size(0)

    def fill(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data

    dis_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    dist_inputs = fill(dist_inputs, dis_mat)
    labels_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    grid_labels = fill(grid_labels, labels_mat)
    mask2d_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.bool)
    grid_mask2d = fill(grid_mask2d, mask2d_mat)
    sub_mat = torch.zeros((batch_size, max_tok, max_pie), dtype=torch.bool)
    pieces2word = fill(pieces2word, sub_mat)

    return bert_inputs,  grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text


class RelationDataset(Dataset):
    def __init__(self, bert_inputs,grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text):
        self.bert_inputs = bert_inputs
        self.grid_labels = grid_labels
        self.grid_mask2d = grid_mask2d
        self.pieces2word = pieces2word
        self.dist_inputs = dist_inputs
        self.sent_length = sent_length
        self.entity_text = entity_text


    def __getitem__(self, item):
        return torch.LongTensor(self.bert_inputs[item]), \
               torch.LongTensor(self.grid_labels[item]), \
               torch.LongTensor(self.grid_mask2d[item]), \
               torch.LongTensor(self.pieces2word[item]), \
               torch.LongTensor(self.dist_inputs[item]), \
               self.sent_length[item], \
               self.entity_text[item], \

    def __len__(self):
        return len(self.bert_inputs)

def split_gaz(gaz):
    """
    Args:
        gaz: List with structure [[skip_input]], where skip_input is a list like:
             [ [[1824], [2]], [], [[4640, 4641], [3, 2]], ... ]
             It may also optionally include volatile_flag as gaz[1].
    Returns:
        [skip_input, volatile_flag]
    """
    skip_input = gaz[0]
    volatile_flag = gaz[1] if len(gaz) > 1 else False
    return [skip_input, volatile_flag]


def process_bert(data, tokenizer, vocab):
    """
    Process BERT input data, converting raw text data into feature representations suitable for model training.
    
    Parameters:
        data: List containing raw text data, each instance includes 'sentence' and 'ner' fields
        tokenizer: BERT tokenizer, used to convert text to token IDs
        vocab: Vocabulary object, used to convert labels to IDs
    
    Returns:
        bert_inputs: List of BERT input token IDs
        grid_labels: Entity relationship grid labels
        grid_mask2d: 2D grid masks
        pieces2word: Token to word mapping relationships
        dist_inputs: Word distance features
        sent_length: List of sentence lengths
        entity_text: Entity text collections
    """
    bert_inputs = []
    gazs_list = []
    grid_labels = []
    grid_mask2d = []
    dist_inputs = []
    entity_text = []
    pieces2word = []
    sent_length = []

    for index, instance in enumerate(data):
        if len(instance['sentence']) == 0:
            continue

        tokens = [tokenizer.tokenize(word) for word in instance['sentence']]
        pieces = [piece for pieces in tokens for piece in pieces]
        _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
        _bert_inputs = np.array([tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])

        length = len(instance['sentence'])
        _grid_labels = np.zeros((length, length), dtype=np.int)
        _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=np.bool)
        _dist_inputs = np.zeros((length, length), dtype=np.int)
        _grid_mask2d = np.ones((length, length), dtype=np.bool)

        if tokenizer is not None:
            start = 0
            for i, pieces_item in enumerate(tokens):
                if len(pieces_item) == 0:
                    continue
                indices = list(range(start, start + len(pieces_item)))
                _pieces2word[i, indices[0] + 1:indices[-1] + 2] = 1
                start += len(pieces_item)

        for k in range(length):
            _dist_inputs[k, :] += k
            _dist_inputs[:, k] -= k

        for i in range(length):
            for j in range(length):
                if _dist_inputs[i, j] < 0:
                    _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
                else:
                    _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
        _dist_inputs[_dist_inputs == 0] = 19

        for entity in instance["ner"]:
            index_list = entity["index"]
            for i in range(len(index_list)):
                if i + 1 >= len(index_list):
                    break
                _grid_labels[index_list[i], index_list[i + 1]] = 1
            _grid_labels[index_list[-1], index_list[0]] = vocab.label_to_id(entity["type"])

        _entity_text = set([utils.convert_index_to_text(e["index"], vocab.label_to_id(e["type"]))
                            for e in instance["ner"]])

        sent_length.append(length)
        bert_inputs.append(_bert_inputs)
        grid_labels.append(_grid_labels)
        grid_mask2d.append(_grid_mask2d)
        dist_inputs.append(_dist_inputs)
        pieces2word.append(_pieces2word)
        entity_text.append(_entity_text)


    return bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_text


def fill_vocab(vocab, dataset):
    entity_num = 0
    for instance in dataset:
        for entity in instance["ner"]:
            vocab.add_label(entity["type"])
        entity_num += len(instance["ner"])
    return entity_num


def build_gaz_file(gaz, gaz_file, skip_first_row=False, separator=" "):
    ## build gaz file,initial read gaz embedding file
    if gaz_file:
        with open(gaz_file, 'r', encoding='utf-8') as file:
            i = 0
            for line in tqdm(file):
                if i == 0:
                    i = i + 1
                    if skip_first_row:
                        _ = line.strip()
                        continue
                fin = line.strip().split(separator)[0]
                if fin:
                    gaz.insert(fin, "one_source")
        print("Load gaz file: ", gaz_file, " total size:", gaz.size())
    else:
        print("Gaz file is None, load nothing")


def read_instance(x, gaz, gaz_alphabet, max_sent_length=200):
    instance_texts = []
    for idx in range(len(x)):
        chars = x[idx]['sentence']
        if ((max_sent_length < 0) or len(chars) < max_sent_length) and (len(chars) > 0):
            gazs = []
            gaz_ids = []
            s_length = len(x[idx]['sentence'])
            for i in range(s_length):
                matched_list = gaz.enumerateMatchList(chars[i:])
                matched_length = [len(a) for a in matched_list]
                gazs.append(matched_list)
                matched_id = [gaz_alphabet.get_index(entity) for entity in matched_list]
                if matched_id:
                    gaz_ids.append([matched_id, matched_length])
                else:
                    gaz_ids.append([])
            x[idx]['word'] = gaz_ids


def load_data_bert(config):
    with open('./data/data/{}/train.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('./data/data/{}/dev.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    with open('./data/data/{}/test.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(config.bert_name, cache_dir="./cache/")

  
    vocab = Vocabulary()

    train_ent_num = fill_vocab(vocab, train_data)
    dev_ent_num = fill_vocab(vocab, dev_data)
    test_ent_num = fill_vocab(vocab, test_data)

    table = pt.PrettyTable([config.dataset, 'sentences', 'entities'])
    table.add_row(['train', len(train_data), train_ent_num])
    table.add_row(['dev', len(dev_data), dev_ent_num])
    table.add_row(['test', len(test_data), test_ent_num])
    config.logger.info("\n{}".format(table))

    config.label_num = len(vocab.label2id)
    config.vocab = vocab
    print(vocab.label2id)
    train_dataset = RelationDataset(*process_bert(train_data, tokenizer, vocab))
    dev_dataset = RelationDataset(*process_bert(dev_data, tokenizer, vocab))
    test_dataset = RelationDataset(*process_bert(test_data, tokenizer, vocab))
    return (train_dataset, dev_dataset, test_dataset), (train_data, dev_data, test_data)

