import math
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class RelationPrediction(nn.Module):
    def __init__(self, config):
        super(RelationPrediction, self).__init__()
        self.config = config
        target_size = config.rel_label
        self.embed = nn.Embedding(config.words_num, config.words_dim)
        if config.train_embed == False:
            self.embed.weight.requires_grad = False
        if config.relation_prediction_mode.upper() == "GRU":
            self.gru = nn.GRU(input_size=config.input_size,
                               hidden_size=config.hidden_size,
                               num_layers=config.num_layer,
                               dropout=config.rnn_dropout,
                               bidirectional=True)
            self.dropout = nn.Dropout(p=config.rnn_fc_dropout)
            self.relu = nn.ReLU()
            self.hidden2tag = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
                nn.BatchNorm1d(config.hidden_size * 2),
                self.relu,
                self.dropout,
                nn.Linear(config.hidden_size * 2, target_size)
            )
        if config.relation_prediction_mode.upper() == "LSTM":
            self.lstm = nn.LSTM(input_size=config.input_size,
                               hidden_size=config.hidden_size,
                               num_layers=config.num_layer,
                               dropout=config.rnn_dropout,
                               bidirectional=True)
            self.dropout = nn.Dropout(p=config.rnn_fc_dropout)
            self.relu = nn.ReLU()
            self.hidden2tag = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
                nn.BatchNorm1d(config.hidden_size * 2),
                self.relu,
                self.dropout,
                nn.Linear(config.hidden_size * 2, target_size)
            )
        if config.relation_prediction_mode.upper() == "CNN":
            input_channel = 1
            Ks = 3
            self.conv1 = nn.Conv2d(input_channel, config.output_channel, (2, config.words_dim), padding=(1, 0))
            self.conv2 = nn.Conv2d(input_channel, config.output_channel, (3, config.words_dim), padding=(2, 0))
            self.conv3 = nn.Conv2d(input_channel, config.output_channel, (4, config.words_dim), padding=(3, 0))
            self.dropout = nn.Dropout(config.cnn_dropout)
            self.fc1 = nn.Linear(Ks * config.output_channel, target_size)


    def forward(self, x):
        # x = (sequence length, batch_size, dimension of embedding)
        text = x.text
        batch_size = text.size()[1]
        x = self.embed(text)
        if self.config.relation_prediction_mode.upper() == "LSTM":
            # # h0 / c0 = (layer*direction, batch_size, hidden_dim)
            # if self.config.cuda:
            #     h0 = Variable(torch.zeros(self.config.num_layer * 2, batch_size,
            #                               self.config.hidden_size).cuda())
            #     c0 = Variable(torch.zeros(self.config.num_layer * 2, batch_size,
            #                               self.config.hidden_size).cuda())
            # else:
            #     h0 = Variable(torch.zeros(self.config.num_layer * 2, batch_size,
            #                               self.config.hidden_size))
            #     c0 = Variable(torch.zeros(self.config.num_layer * 2, batch_size,
            #                               self.config.hidden_size))
            # output = (sentence length, batch_size, hidden_size * num_direction)
            # ht = (layer*direction, batch, hidden_dim)
            # ct = (layer*direction, batch, hidden_dim)
            # outputs, (ht, ct) = self.lstm(x, (h0, c0))
            outputs, (ht, ct) = self.lstm(x)
            tags = self.hidden2tag(ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1))  # shape: (batch_size, direction*hidden_dim)
            scores = F.log_softmax(tags, dim=1)
            return scores
        elif self.config.relation_prediction_mode.upper() == "GRU":
            # if self.config.cuda:
            #     h0 = Variable(torch.zeros(self.config.num_layer * 2, batch_size,
            #                               self.config.hidden_size).cuda())
            # else:
            #     h0 = Variable(torch.zeros(self.config.num_layer * 2, batch_size,
            #                               self.config.hidden_size))
            #outputs, ht = self.gru(x, h0)
            outputs, ht = self.gru(x)

            tags = self.hidden2tag(ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1))
            scores = F.log_softmax(tags, dim=1)
            return scores
        elif self.config.relation_prediction_mode.upper() == "CNN":
            x = x.transpose(0, 1).contiguous().unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim)
            x = [F.relu(self.conv1(x)).squeeze(3), F.relu(self.conv2(x)).squeeze(3), F.relu(self.conv3(x)).squeeze(3)]
            # (batch, channel_output, ~=sent_len) * Ks
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling
            # (batch, channel_output) * Ks
            x = torch.cat(x, 1)  # (batch, channel_output * Ks)
            x = self.dropout(x)
            logit = self.fc1(x)  # (batch, target_size)
            scores = F.log_softmax(logit, dim=1)
            return scores
        else:
            print("Unknown Mode")
            exit(1)

class TransformerModel(nn.Module):
    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.config = config
        target_size = config.rel_label
        self.d_model = config.words_dim
        self.embed = nn.Embedding(config.words_num, self.d_model)
        # self.src_embed = nn.Embedding(config.words_num, config.words_dim)
        # self.tgt_embed = nn.Embedding()
        if config.train_embed == False:
            self.embed.weight.requires_grad = False
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(self.d_model, config.transformer_dropout)
        encoderlayer = TransformerEncoderLayer(self.d_model, config.nhead,
                                               config.dim_feedforward, config.transformer_dropout)
        encoder_norm = nn.LayerNorm(self.d_model)

        self.concat = False
        if self.concat == True:
            self.transformer_encoder = TransformerEncoder(encoderlayer,
                                                        config.num_encoder_layers - 1,
                                                        encoder_norm)
            self.last_encoder_layer = TransformerEncoder(encoderlayer, 1, encoder_norm)
        else:
            self.transformer_encoder = TransformerEncoder(encoderlayer,
                                                        config.num_encoder_layers,
                                                        encoder_norm)
        
        self.dropout = nn.Dropout(p=config.transformer_dropout)
        self.relu = nn.ReLU()
        d_model_linear = self.d_model * (1 + self.concat)
        self.hidden2tag = nn.Sequential(
            nn.Linear(d_model_linear, d_model_linear),
            nn.BatchNorm1d(d_model_linear),
            self.relu,
            self.dropout,
            nn.Linear(d_model_linear, target_size)
        )
        # self.decoder = nn.Linear(self.d_model, target_size)
        self.init_weights()

    def _generate_square_subsequent_mask(self, seq_len):
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):
        src = src.text
        if self.src_mask is None or self.src_mask.size(0) != src.size(0):
            device = src.device
            mask = self._generate_square_subsequent_mask(src.size(0)).to(device)
            self.src_mask = mask
        
        src = self.embed(src) #* math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        if self.concat == True:
            output1 = self.transformer_encoder(src)
            output2 = self.last_encoder_layer(output1)
            output = torch.cat([output1, output2], dim=-1)
        else:
            output = self.transformer_encoder(src) # not use src_mask
        # output = self.decoder(output)

        # output of shape (seq_len, batch_size, d_model)
        # output = torch.mean(output, dim=0)
        # output = torch.max(output, dim=0)[0]
        output = output[0,:,:]
        if output.size(0) != src.size(1):
            raise RuntimeError("the result does not match the batch size")
        output = self.hidden2tag(output)
        scores = F.log_softmax(output, dim=1)
        return scores

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        # attention the dimension
        return self.dropout(x)

