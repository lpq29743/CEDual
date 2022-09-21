import os
# import logging 
import argparse
import torch

UNK_idx = 0
PAD_idx = 1
EOS_idx = 2
SOS_idx = 3
USR_idx = 4  # speak state
SYS_idx = 5  # listener state
CLS_idx = 6
LAB_idx = 7 # emotional state

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if (os.cpu_count() > 8):
    USE_CUDA = True
else:
    USE_CUDA = False

parser = argparse.ArgumentParser()
parser.add_argument("--hidden_dim", type=int, default=300)
parser.add_argument("--emb_dim", type=int, default=300)
parser.add_argument("--n_epoch", type=int, default=10000)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--max_grad_norm", type=float, default=2.0)
parser.add_argument("--beam_size", type=int, default=5)
parser.add_argument("--save_path", type=str, default="save/test/")
parser.add_argument("--resume_path", type=str, default="result/")
parser.add_argument("--cuda", action="store_true", default=True)
parser.add_argument('--device_id', dest='device_id', type=str, default="0")
parser.add_argument('--dropout', dest='dropout', type=float, default=0.2)

parser.add_argument("--pointer_gen", action="store_true", default=True)
parser.add_argument("--beam_search", action="store_true")
parser.add_argument("--oracle", action="store_true")
parser.add_argument("--basic_learner", action="store_true", default=True)
parser.add_argument("--project", action="store_true")
parser.add_argument("--emotion_bia", action="store_true")
parser.add_argument("--global_update", action="store_true")
parser.add_argument("--topk", type=int, default=0)
parser.add_argument("--teacher_ratio", type=float, default=1.0)
parser.add_argument("--l1", type=float, default=.0)
parser.add_argument("--softmax", action="store_true")
parser.add_argument("--mean_query", action="store_true")
parser.add_argument("--schedule", type=float, default=0)

parser.add_argument("--large_decoder", action="store_true")
parser.add_argument("--multitask", action="store_true")
parser.add_argument("--is_coverage", action="store_true")
parser.add_argument("--use_oov_emb", action="store_true")
parser.add_argument("--pretrain_emb", action="store_true", default=True)
parser.add_argument("--weight_sharing", action="store_true")
parser.add_argument("--label_smoothing", action="store_true", default=True)
parser.add_argument("--noam", action="store_true", default=True)
parser.add_argument("--universal", action="store_true")
parser.add_argument("--act", action="store_true")
parser.add_argument("--act_loss_weight", type=float, default=0.001)
parser.add_argument("--empathy_label", action="store_true")

parser.add_argument("--emb_file", type=str)


parser.add_argument("--hop", type=int, default=1)
parser.add_argument("--heads", type=int, default=2)
parser.add_argument("--depth", type=int, default=40)
parser.add_argument("--filter", type=int, default=50)

parser.add_argument("--resume_g", action="store_true")
parser.add_argument("--resume_d", action="store_true")
parser.add_argument("--adver_train", action="store_true")
parser.add_argument("--gp_lambda", type=int, default=0.1)
parser.add_argument("--rnn_hidden_dim", type=int, default=300)
parser.add_argument("--d_steps", type=int, default=1)
parser.add_argument("--g_steps", type=int, default=5)
parser.add_argument("--emotion_disc", action="store_true", default=True)
parser.add_argument("--adver_itr_num", type=int, default=100000)
parser.add_argument("--emotion_state_emb", action="store_true")


def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)

arg = parser.parse_args()
print_opts(arg)
large_decoder = arg.large_decoder
emotion_bia = arg.emotion_bia
global_update = arg.global_update
topk = arg.topk
dropout = arg.dropout
l1 = arg.l1
oracle = arg.oracle
beam_search = arg.beam_search
basic_learner = arg.basic_learner
teacher_ratio = arg.teacher_ratio
multitask = arg.multitask
softmax = arg.softmax
mean_query = arg.mean_query
schedule = arg.schedule
hidden_dim= arg.hidden_dim
emb_dim= arg.emb_dim
batch_size= arg.batch_size
lr=arg.lr
beam_size=arg.beam_size
project=arg.project
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=arg.max_grad_norm

USE_CUDA = arg.cuda
device_id = arg.device_id
pointer_gen = arg.pointer_gen
is_coverage = arg.is_coverage
use_oov_emb = arg.use_oov_emb
cov_loss_wt = 1.0
lr_coverage=0.15
eps = 1e-12
epochs = arg.n_epoch

emb_file = arg.emb_file or "vectors/glove.6B.{}d.txt".format(str(emb_dim))
pretrain_emb = arg.pretrain_emb

save_path = arg.save_path

hop = arg.hop
heads = arg.heads
depth = arg.depth
filter = arg.filter

label_smoothing = arg.label_smoothing
weight_sharing = arg.weight_sharing
noam = arg.noam
universal = arg.universal
act = arg.act
act_loss_weight = arg.act_loss_weight
empathy_label = arg.empathy_label

# logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M', filename='save/logs/{}.log'.format(str(model)))
collect_stats = False

resume_path = arg.resume_path
resume_g = arg.resume_g
gp_lambda = arg.gp_lambda
rnn_hidden_dim = arg.rnn_hidden_dim
d_steps = arg.d_steps
g_steps = arg.g_steps
adver_train = arg.adver_train
emotion_disc = arg.emotion_disc
resume_d = arg.resume_d
adver_itr_num = arg.adver_itr_num
emotion_state_emb = arg.emotion_state_emb

