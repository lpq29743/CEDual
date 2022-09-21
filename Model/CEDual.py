import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.common_layer import EncoderLayer, DecoderLayer, MultiHeadAttention, Conv, PositionwiseFeedForward, LayerNorm , _gen_bias_mask ,_gen_timing_signal, share_embedding, LabelSmoothing, NoamOpt, _get_attn_subsequent_mask
import numpy as np
import math
from utils import config
import random
import os
import time
from copy import deepcopy
from sklearn.metrics import accuracy_score
import pdb

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

class Encoder(nn.Module):
    """
    A Transformer Encoder module. 
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0, 
                 attention_dropout=0.0, relu_dropout=0.0, use_mask=False, universal=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """
        
        super(Encoder, self).__init__()
        self.universal = universal
        self.num_layers = num_layers
        self.input_dropout = nn.Dropout(input_dropout)
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        
        if(self.universal):  
            ## for t
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params =(hidden_size, 
                 total_key_depth or hidden_size,
                 total_value_depth or hidden_size,
                 filter_size, 
                 num_heads, 
                 _gen_bias_mask(max_length) if use_mask else None,
                 layer_dropout, 
                 attention_dropout, 
                 relu_dropout)
        
        if(self.universal):
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])
        
        self.layer_norm = LayerNorm(hidden_size)
        
        if(config.act):
            self.act_fn = ACT_basic(hidden_size)
            self.remainders = None
            self.n_updates = None

    def forward(self, inputs, mask):
        #Add input dropout
        x = self.input_dropout(inputs)
        
        # Project to hidden size
        x = self.embedding_proj(x)
        
        if(self.universal):
            if(config.act):
                x, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.enc, self.timing_signal, self.position_signal, self.num_layers)
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1,inputs.shape[1],1).type_as(inputs.data)
                    x = self.enc(x, mask=mask)
                y = self.layer_norm(x)
        else:
            # Add timing signal
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
            
            for i in range(self.num_layers):
                x = self.enc[i](x, mask)
        
            y = self.layer_norm(x)
        return y

class MulDecoder(nn.Module):
    def __init__(self, expert_num,  embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0, 
                 attention_dropout=0.0, relu_dropout=0.0):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """
        
        super(MulDecoder, self).__init__()
        self.num_layers = num_layers
        self.mask = _get_attn_subsequent_mask(max_length)
        self.input_dropout = nn.Dropout(input_dropout)
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)

        params =(hidden_size, 
                 total_key_depth or hidden_size,
                 total_value_depth or hidden_size,
                 filter_size, 
                 num_heads, 
                 _gen_bias_mask(max_length), # mandatory
                 layer_dropout, 
                 attention_dropout, 
                 relu_dropout)
        self.content_dec1 = DecoderLayer(*params)
        self.emotion_dec1 = DecoderLayer(*params)
        self.content_dec2 = DecoderLayer(*params)
        self.emotion_dec2 = DecoderLayer(*params)
        self.output_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.layer_norm = LayerNorm(hidden_size)

    def forward(self, inputs, content_output, emotion_output, mask):
        mask_src, mask_trg = mask
        dec_mask = torch.gt(mask_trg.bool() + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)].bool(), 0)

        #Add input dropout
        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)
        # Add timing signal
        x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
        # First content then emotion
        # content_out1, _, attn_dist, _ = self.content_dec1((x, content_output, [], (mask_src,dec_mask)))
        # y, _, attn_dist, _ = self.emotion_dec1((content_out1, emotion_output, [], (mask_src,dec_mask)))
        # First emotion then content
        emotion_out2, _, attn_dist, _ = self.emotion_dec2((x, emotion_output, [], (mask_src,dec_mask)))
        y, _, attn_dist, _ = self.content_dec2((emotion_out2, content_output, [], (mask_src,dec_mask)))

        # Final layer normalization
        y = self.layer_norm(y)
        return y, attn_dist

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x, attn_dist=None, enc_batch_extend_vocab=None, extra_zeros=None, temp=1, beam_search=False, attn_dist_db=None):
        logit = self.proj(x)
        return F.log_softmax(logit,dim=-1)

class CEDual(nn.Module):

    def __init__(self, vocab, decoder_number, model_file_path=None, is_eval=False, load_optim=False):
        super(CEDual, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.n_words
        self.decoder_number = decoder_number

        self.embedding = share_embedding(self.vocab, config.pretrain_emb)
        self.encoder = Encoder(config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads, 
                                total_key_depth=config.depth, total_value_depth=config.depth,
                                filter_size=config.filter,universal=config.universal) 

        self.decoder = MulDecoder(decoder_number, config.emb_dim, config.hidden_dim, num_layers=config.hop, num_heads=config.heads, 
                                    total_key_depth=config.depth, total_value_depth=config.depth,
                                    filter_size=config.filter)
        self.generator = Generator(config.hidden_dim, self.vocab_size)
        if config.weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.generator.proj.weight = self.embedding.lut.weight
        self.criterion = nn.NLLLoss(ignore_index=config.PAD_idx)
        if (config.label_smoothing):
            self.criterion = LabelSmoothing(size=self.vocab_size, padding_idx=config.PAD_idx, smoothing=0.1)
            self.criterion_ppl = nn.NLLLoss(ignore_index=config.PAD_idx)
        
        params =(config.hidden_dim, 
                 config.hidden_dim,
                 config.hidden_dim,
                 config.filter,
                 config.heads)
        self.content_proj = EncoderLayer(*params)
        self.emotion_proj = EncoderLayer(*params)
        # self.content_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        # self.emotion_proj = nn.Linear(config.hidden_dim, config.hidden_dim)        
        self.emotion_classifier = nn.Linear(config.hidden_dim, decoder_number)

        self.tanh = nn.Tanh()
        self.activation = nn.Softmax(dim=1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr)
        if(config.noam):
            self.optimizer = NoamOpt(config.hidden_dim, 1, 8000, torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        if model_file_path is not None:
            print("loading weights")
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'])
            self.decoder_key.load_state_dict(state['decoder_key_state_dict']) 
            self.generator.load_state_dict(state['generator_dict'])
            self.embedding.load_state_dict(state['embedding_dict'])
            if (load_optim):
                self.optimizer.load_state_dict(state['optimizer'])
            self.eval()

        self.model_dir = config.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def save_model(self, running_avg_ppl, iter, f1_g,f1_b,ent_g,ent_b):
        state = {
            'iter': iter,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'decoder_key_state_dict': self.decoder_key.state_dict(),
            'generator_dict': self.generator.state_dict(),
            'embedding_dict': self.embedding.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_ppl
        }
        model_save_path = os.path.join(self.model_dir, 'model_{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(iter,running_avg_ppl,f1_g,f1_b,ent_g,ent_b) )
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def train_one_batch(self, batch, iter, train=True):
        enc_batch = batch["context_batch"]
        enc_ext_batch = batch["context_ext_batch"]
        oovs = batch["oovs"]
        max_oov_length = len(sorted(oovs, key=lambda i: len(i), reverse=True)[0])

        dec_batch = batch["target_batch"]
        dec_ext_batch = batch["target_ext_batch"]

        if config.noam:
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()
        ## Encode
        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mask = self.embedding(batch["mask_context"])  # (bsz, src_len)->(bsz, src_len, embed_size)
        encoder_outputs = self.encoder(self.embedding(enc_batch)+emb_mask,mask_src)

        content_outputs = self.tanh(self.content_proj(encoder_outputs))
        emotion_outputs = self.tanh(self.emotion_proj(encoder_outputs))
        
        
        # Decode 
        sos_token = torch.LongTensor([config.SOS_idx] * enc_batch.size(0)).unsqueeze(1)
        if config.USE_CUDA: sos_token = sos_token.cuda()
        dec_batch_shift = torch.cat((sos_token,dec_batch[:, :-1]),1)

        mask_trg = dec_batch_shift.data.eq(config.PAD_idx).unsqueeze(1)
        pre_logit, attn_dist = self.decoder(self.embedding(dec_batch_shift), content_outputs, emotion_outputs, (mask_src,mask_trg))
        ## compute output dist
        logit = self.generator(pre_logit, attn_dist, enc_ext_batch if config.pointer_gen else None, max_oov_length, attn_dist_db=None)
        ## loss: NNL if ptr else Cross entropy
        if(train and config.schedule>10):
            if(random.uniform(0, 1) <= (0.0001 + (1 - 0.0001) * math.exp(-1. * iter / config.schedule))):
                config.oracle=True
            else:
                config.oracle=False
        
        loss_generation = self.criterion(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1))
        # Disentangle
        content_rep = torch.mean(content_outputs, dim=1)
        content_prob = self.activation(self.emotion_classifier(content_rep))
        loss_content = torch.sum(content_prob * torch.log(content_prob))
        emotion_rep = torch.mean(emotion_outputs, dim=1)
        emotion_prob = self.emotion_classifier(emotion_rep)
        loss_emotion = nn.CrossEntropyLoss(reduction='sum')(emotion_prob, batch['emotion_label'])
        
        loss = loss_generation + loss_content + loss_emotion

        pred_emotion = np.argmax(emotion_prob.detach().cpu().numpy(), axis=1)
        emotion_acc = accuracy_score(batch["emotion_label"].cpu().numpy(), pred_emotion)

        if(config.label_smoothing): 
            loss_ppl = self.criterion_ppl(logit.contiguous().view(-1, logit.size(-1)), dec_batch.contiguous().view(-1)).item()
        
        if(train):
            loss.backward()
            self.optimizer.step()

        if config.label_smoothing:
            return loss_ppl, math.exp(min(loss_ppl, 100)), loss_emotion.item(), emotion_acc
        else:
            return loss.item(), math.exp(min(loss.item(), 100)), 0, 0

    def compute_act_loss(self,module):    
        R_t = module.remainders
        N_t = module.n_updates
        p_t = R_t + N_t
        avg_p_t = torch.sum(torch.sum(p_t,dim=1)/p_t.size(1))/p_t.size(0)
        loss = config.act_loss_weight * avg_p_t.item()
        return loss

    def decoder_greedy(self, batch, max_dec_step=30):
        enc_batch = batch["context_batch"]
        enc_ext_batch = batch["context_ext_batch"]
        oovs = batch["oovs"]
        max_oov_length = len(sorted(oovs, key=lambda i: len(i), reverse=True)[0])

        mask_src = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)
        emb_mask = self.embedding(batch["mask_context"])
        encoder_outputs = self.encoder(self.embedding(enc_batch)+emb_mask,mask_src)
        content_outputs = self.tanh(self.content_proj(encoder_outputs))
        emotion_outputs = self.tanh(self.emotion_proj(encoder_outputs))
 
        ys = torch.ones(1, 1).fill_(config.SOS_idx).long()
        if config.USE_CUDA:
            ys = ys.cuda()
        mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)
        decoded_words = []

        for i in range(max_dec_step+1):
            out, attn_dist = self.decoder(self.embedding(ys), content_outputs, emotion_outputs, (mask_src,mask_trg))
            logit = self.generator(out, attn_dist, enc_ext_batch if config.pointer_gen else None, max_oov_length,
                                   attn_dist_db=None)
            #logit = F.log_softmax(logit,dim=-1) #fix the name later
            _, next_word = torch.max(logit[:, -1], dim = 1)
            decoded_words.append(['<EOS>' if ni.item() == config.EOS_idx else self.vocab.index2word[ni.item()] for ni in next_word.view(-1)])
            next_word = next_word.data[0]
            if config.USE_CUDA:
                ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word).cuda()], dim=1)
                ys = ys.cuda()
            else:
                ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word)], dim=1)
            mask_trg = ys.data.eq(config.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<EOS>': break
                else: st+= e + ' '
            sent.append(st)
        return sent

