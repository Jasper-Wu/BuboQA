# Strong Baselines for Simple Question Answering over Knowledge Graphs with and without Neural Networks

# Configuration 

wsl2 fail

docker

- should successfully install wsl2 first and then setup the wsl2 kernel in docker
- Failed to create or configure Hyper-V VM: Sequence contains no elements..

environment

```shell
conda activate i2dl
import torch
print(torch.__version__)

pip install torchtext
pip install nltk
pip install fuzzywuzzy

# download nltk data in python environment
import nltk
nltk.download()    # folder: C:\nltk_data or /usr/share/nltk_data
# in wsl 
python -m nltk.downloader all  # location: /home/jasper/nltk_data
# test that data has been installed
from nltk.corpus import brown
brown.words()
```

wget for windows    

- https://eternallybored.org/misc/wget/
- in C:\Windows\System32 or add path to `PATH` variable in environment variable 
- `wget -h` check if successfully install

running `sh setup.sh` in windows

- using wsl

  error like `$'\r': command not found`

  `$'\r'$`是回车符（CR）的表示形式，是DOS和Windows行尾（CR LF）的一部分，但在Unix的行尾（LF）不存在

  ```bash
  # in WSL
  sudo apt-get install dos2unix
  dos2unix [file]
  man dos2unix
  # for example
  wsl dos2unix deploy.sh
  wsl ./deploy.sh
  ```

  in Visual Studio Code change the End of Line Sequence (from `CRLF` to `LF`), so that can be recognized by wsl

  but in order to run python file, we need some packages, so maybe better first install anaconda (in env `py37`)

  (题外话：安装anaconda创建环境之后里面的python版本是错误的，安装pytorch之后也无法`import torch`，后来是关机重开解决的，还有一个是之前设置了python的alias，感觉应该没有关系)

  after configuring the environment (pytorch cpuonly version, no nltk_data), it seems that everything works well (no error report)

  have to mention that the `num keys`, `key-value pairs` are different to the result in Colab

  it fails again, it seems that `/data/sq_glove300d.pt` file has some problem, which leads to error like `_pickle.UnpicklingError: pickle data was truncated`, later on I download the same file processed in Colab, and it works well

- using Git Bash

  still get some error, in `scripts/fetch_dataset.sh`, `echo "\n\nTrimming the names file for subset 2M...\n"`

  see the screenshots

- using Colab

  ```
  # set the `NLTK_DATA` environment variable
  import os
  os.environ['NLTK_DATA']="/content/drive/My Drive/BuboQA/nltk_data"
  ```



- 



# Component


- SimpleQuestions Dataset: https://research.fb.com/downloads/babi/
- Freebase: https://developers.google.com/freebase/guide/basic_concepts
- torchtext
- Embedding: GloVe Embeddings (Pennington et al., 2014)




- Entity Detection

  - Recurrent Neural Networks: bi-directional LSTM and GRU
  - Conditional Random Fields (CRFs)
- Entity Linking
- Relation Prediction

  - RNNs: BiLSTM and BiGRU
  - CNNs
  - Logistic Regression
    - tf-idf
    - word embedding + relation words ???


- Evidence Integration 



## Transformer

> Induction, deriving the function from the given data. Deduction, deriving the values of the given function for points of interest. Transduction, deriving the values of the unknown function for points of interest from the given data.    -- Page 169, [The Nature of Statistical Learning Theory](http://amzn.to/2uvHt5a), 1995

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [What Are Transformer Models in Machine Learning?](https://lionbridge.ai/articles/what-are-transformer-models-in-machine-learning/)

Code

- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html): [GitHub](https://github.com/harvardnlp/annotated-transformer)
- [Sequence-to-Sequence Modeling with nn.Transformer and torchtext](https://pytorch.org/tutorials/beginner/transformer_tutorial.html) (PyTorch Tutorial)
- [State-of-the-art Natural Language Processing for PyTorch and TensorFlow 2.0](https://github.com/huggingface/transformers)
- [Transformers in NLP: Creating a Translator Model from Scratch](https://lionbridge.ai/articles/transformers-in-nlp-creating-a-translator-model-from-scratch/)

Hyperparameters: Entity Detection 

- dropout (0.1? or 0.5 in pytorch tutorial): `dropout=0.3` seems not work 
- multi-head, word embedding dimension, num_encoder_layer
- `src_mask`: worse than without `src_mask`
- weight initialization
- optimizer
  - SGD with `lr=5` and scheduler not work (copy from pytorch tutorial)
  - original Adam or Adam with specifying $\beta$ make no difference
  - Adam with `lr=1e-3` fail
  - with warmup
- classification layer 
  - seems no much difference
- $\times \sqrt{d_{model}}$ after embedding: without this step gives better result (just once run, no guarantee)

Hyperparameter: Relation Prediction

- combine `seq_len` data into one piece of data for subsequent classification: max, concatenate, BERT

  - mean

  - max: if choose one vector from these (`seq_len`) vectors, what is the matric to calculate max; or max operation to get the max value in each dimension

    what about the gradient? do I need to concerned about the gradient calculation?

  - concatenate: but the `seq_len` is different from batch to batch, so it seems not a good idea, not able to determine a fixed size input for subsequent layer (is it possible to fill in padding to extend each to size `max_seq_len`?)

  - BERT: with `<cls>` tag

- $\times \sqrt{d_{model}}$ after embedding: again without this step gives better result

- `dim_feedforward=512, num_encoder_layers=9` worse than `dim_feedforward=2048, num_encoder_layers=6`

## BERT

- [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](http://jalammar.github.io/illustrated-bert/)
- https://github.com/google-research/bert
- [pre-trained BERT model](https://github.com/google-research/bert#pre-trained-models)
- [A Hands-On Guide To Text Classification With Transformer Models (XLNet, BERT, XLM, RoBERTa)](https://towardsdatascience.com/https-medium-com-chaturangarajapakshe-text-classification-with-transformer-models-d370944b50ca) with [notebook in GitHub](https://github.com/ThilinaRajapakse/pytorch-transformers-classification)



Size Comparison

|        | LSTM                                  | transformer |
| ------ | ------------------------------------- | ----------- |
| input  | (seq_len, batch_size, dimen_embed)    |             |
| output | (seq_len, batch_size, hid_size*direc) |             |
|        |                                       |             |




- GPU
  - GTX 950M: 185s/k_iteration VS. Colab: 34s/k_iteration

# Run

## Result

- **Entity Detection** 

  | metric: F1     | BiLSTM           | Transformer |
  | -------------- | ---------------- | ----------- |
  | validation set | 93.1 [92.8 93.4] | 94.15       |
  | test set       |                  | 93.78       |

  

- **Entity Linking**

  Validation Set (from paper)

  | R$@N$ | BiLSTM           | CRF  | Transformer |
  | ----- | ---------------- | ---- | ----------- |
  | 1     | 67.8 [67.5 68.0] | 66.6 | 68.1        |
  | 5     | 82.6 [82.3 82.7] | 81.3 | 82.8        |
  | 20    | 88.7 [88.5 88.8] | 87.4 | 88.9        |
  | 50    | 91.0 [90.8 91.1] | 89.8 | 91.3        |

  Test Set (from `README.md`)

  | R$@N$ | BiLSTM | CRF  | Transformer |
  | ----- | ------ | ---- | ----------- |
  | 1     | 65.0   | 63.7 | 66.6        |
  | 5     | 79.8   | 78.3 | 81.4        |
  | 20    | 85.8   | 84.0 | 88.0        |
  | 50    | 88.4   | 86.7 | 90.6        |

  

- **Relation Prediction**

  Validation (from paper)

  | Model       | R@1              | R@5              |
  | ----------- | ---------------- | ---------------- |
  | BiGRU       | 82.3 [82.0 82.5] | 95.9 [95.7 96.1] |
  | CNN         | 82.8 [82.5 82.9] | 95.8 [95.7 96.1] |
  | Transformer | 81.0             | 93.2             |

  Test (from `README.md`)

  | Model       | R@1   | R@5   |
  | ----------- | ----- | ----- |
  | BiGRU       | 81.77 | 95.74 |
  | CNN         | 82.12 | 95.66 |
  | Transformer | 79.93 | 92.84 |

  model alternative: mean, max, cls_tag, end_tag, fix_embed, concatenation

  

- Evidence Integration

  Validation Set (suspect )

  | Entity      | Relation    | Acc.             |
  | ----------- | ----------- | ---------------- |
  | BiLSTM      | BiGRU       | 74.9 [74.6 75.1] |
  | BiLSTM      | CNN         | 74.7 [74.5 74.9] |
  | transformer | transformer | 70.2             |

  Test

  | Entity      | Relation    | Acc.  |
  | ----------- | ----------- | ----- |
  | BiLSTM      | BiGRU       | 73.15 |
  | BiLSTM      | CNN         | 73.19 |
  | transformer | transformer | 68.16 |

## Error

entity detection

- `top_retrieval.py`  error:

  ```python
  results_file.write("{}\t{}\t{}\n".format(" ".join(question), " ".join(label), " ".join(gold)))
  UnicodeEncodeError: 'gbk' codec can't encode character '\u010d' in position 20: illegal multibyte sequence
  ```

  

## Question

- `entity_detection/nn/evaluation.py` function `get_span` why use `index2tag[label[k]][0]`; what happens if `type == True`, and `index2tag[label[k]][2:]`
- entity detection: training use accuracy, development set use precision, recall, and f1, different metric
- in `entity detection` task, use `train, dev, test` to build the vocabulary, but in `relation prediction` task, relation just use `train, dev` sets 
- `model = torch.load(args.trained_model, map_location=lambda storage, location: storage.cuda(args.gpu))`
- `relation_prediction/nn/top_retrieval.py` `example = data_batch.dataset.examples[index]`

# Related Papers

## Simple Recurrent Neural Networks Work!

*No Need to Pay Attention: Simple Recurrent Neural Networks Work! (for Answering "Simple" Questions)*

- Entity Detection
- Relation Prediction
- Entity Linking 
- Answer Selection 



# Code Excerpt

```python
'''
from: Torchtext使用教程 https://www.jianshu.com/p/71176275fdc5
'''
import spacy
import torch
from torchtext import data, datasets
from torchtext.vocab import Vectors
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_csv('train.tsv', sep='\t')
test = pd.read_csv('test.tsv', sep='\t')

train, val = train_test_split(data, test_size=0.2)
train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)

spacy_en = spacy.load('en')

def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True) # there are many parameters for Field
LABEL = data.Field(sequential=False, use_vocab=False)

train, val = data.TabularDataset.splits(path='.', train='train.csv', validation='val.csv', format='csv', skip_header=True, fields=[('PhraseId', None),('SentenceId', None), ('Phrase', TEXT), ('Sentiment', LABEL)])
test = data.TabularDataset('test.tsv', format='tsv', skip_header=True, fields=[('PhraseId', None), ('SentenceId', None), ('Phrase', TEXT)])

print(train[5]) # <torchtext.data.example.Example object at ...>
print(train[5].__dict__.keys()) # dict_keys(['Phrase', 'Sentiment'])
print(train[5].Phrase, train[0].Sentiment) # ['movie', 'pool'] 2
# not yet transform to number 

TEXT.build_vocab(train, vectors='glove.6B.100d')
# 当corpus中有token在vectors中不存在时的初始化方式
TEXT.vocab.vectors.unk_init = init.xavier_uniform

print(TEXT.vocab.itos[1510]) # bore
print(TEXT.vocab.stoi['bore']) # 1510
print(TEXT.vocab.vectors.shape) # torch.Size([15422, 100])
word_vec = TEXT.vocab.vectors[TEXT.vocab.stoi['bore']]
print(word_vec.shape) # torch.Size([100])
print(word_vec)

train_iter = data.BucketIterator(train, batch_size=128, sort_key=lambda x: len(x.Phrase), shuffle=True, device=DEVICE)
val_iter = data.BucketIterator(val, batch_size=128, sort_key=lambda x: len(x.Phrase), shuffle=True, device=DEVICE)
test_iter = data.BucketIterator(dataset=test, batch_size=128, train=False, sort=False, device=DEVICE)

batch = next(iter(train_iter))
data = batch.Phrase
label = batch.Sentiment
print(batch.Phrase.shape) # torch.Size([41, 128])
print(batch.Phrase)
# or 
for batch in train_iter:
    data = batch.Phrase
    label = batch.Sentiment
    
class Enet(nn.Module):
    def __init__(self):
        super(Enet, self).__init__()
        self.embedding
        self.lstm
        self.linear
        
    def forward(self, x):
        ...
        return out
    
model = Enet()
model.embedding.weight.data.copy_(TEXT.vocab.vectors)
model.to(DEVICE)

optimizer = optim.Adam(model.parameters())
n_epoch = 20
best_val_acc = 0

for epoch in range(n_epoch):
    for batch_idx, batch in enumerate(train_iter):
        ...
        
    for batch_idx, batch in enumerate(val_iter):
```

**contiguous**

```python
'''
*PyTorch中的contiguous*: https://zhuanlan.zhihu.com/p/64551412
'''
>>>t = torch.arange(12).reshape(3,4)
>>>t
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
>>>t.stride()
(4, 1)
>>>t2 = t.transpose(0,1)
>>>t2
tensor([[ 0,  4,  8],
        [ 1,  5,  9],
        [ 2,  6, 10],
        [ 3,  7, 11]])
>>>t2.stride()
(1, 4)
>>>t.data_ptr() == t2.data_ptr() # 底层数据是同一个一维数组
True
>>>t.is_contiguous(),t2.is_contiguous() # t连续，t2不连续
(True, False)

>>>t3 = t2.contiguous()
>>>t3
tensor([[ 0,  4,  8],
        [ 1,  5,  9],
        [ 2,  6, 10],
        [ 3,  7, 11]])
>>>t3.data_ptr() == t2.data_ptr() # 底层数据不是同一个一维数组
False

# transpose(), permute(), expand(), view(), narrow()不修改底层一维数组，而是改变元数据，如offset和stride
```

```python
class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout):
        super(TransformerModel, self).__init__()
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()
        ...
        
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0,1)
        mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
        return mask
    
    def init_weights():
        pass
    
    def forward(self, src):
        if self.src_mask is None:
            pass
        src = self.encoder(src) * math.sqrt(self.ninp)	# easy to forget this step
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output
    
class PositionalEncoding(nn.Module):
    def __init__():
        # pay attention to the dimension
        pass
    def forward():
        pass
    
```

