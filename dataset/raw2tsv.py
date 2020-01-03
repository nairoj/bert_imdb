import os
import spacy
import random
import csv
import torchtext


def train_data2tsv(path):
    examples = []
    d = {'pos': 1, 'neg': 0}
    for label in ['pos', 'neg']:
        label_idx = d[label]
        filepath = os.path.join(path, label)
        filelist = os.listdir(filepath)

        for fname in filelist:
            with open(os.path.join(filepath, fname), 'r', encoding="utf-8") as f:
                print(fname)
                text = f.readline()
                examples.append([tokenize(text), label_idx])
    random.shuffle(examples)
    train_data_len = len(examples) // 10 * 9
    train_data = examples[:train_data_len]
    dev_data = examples[train_data_len:]
    with open('train.tsv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(train_data)

    with open('dev.tsv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(dev_data)


def test_data2tsv(path):
    examples = []
    d = {'pos': 1, 'neg': 0}
    for label in ['pos', 'neg']:
        label_idx = d[label]
        filepath = os.path.join(path, label)
        filelist = os.listdir(filepath)

        for fname in filelist:
            with open(os.path.join(filepath, fname), 'r', encoding="utf-8") as f:
                print(fname)
                text = f.readline()
                examples.append([tokenize(text), label_idx])

    random.shuffle(examples)
    with open('test.tsv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(examples)


def tokenize(txt):
    tokens = tokenizer(txt)
    return ' '.join([t.text for t in tokens if not t.is_stop])


if __name__ == '__main__':
    # 下载imdb数据,要看网络情况
    # print('下载IMDB数据')
    # torchtext.datasets.IMDB.download('./')
    
    # 把torchtext下载下来的IMDB数据转换成bert微调脚本的可用格式
    tokenizer = spacy.load('en')
    tokenizer.vocab['\t'].is_stop = True
    
    print('开始转换测试集')
    path = './aclImdb/test'
    test_data2tsv(path)
    
    print('开始转换训练集和验证集')
    path = './aclImdb/train'
    train_data2tsv(path)
