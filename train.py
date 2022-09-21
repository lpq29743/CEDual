import os
from tensorboardX import SummaryWriter
from copy import deepcopy
from torch.nn.init import xavier_uniform_
import numpy as np
import random
from utils.data_loader import prepare_data_seq
from utils.common import *

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

from utils.data_reader import Lang
from Model.CEDual import CEDual

os.environ["CUDA_VISOBLE_DEVICES"] = config.device_id
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
if torch.cuda.is_available():
    torch.cuda.set_device(int(config.device_id))

if __name__ == '__main__':
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    data_loader_tra, data_loader_val, data_loader_tst, vocab, program_number = prepare_data_seq(batch_size=config.batch_size) 
    
    model = CEDual(vocab, decoder_number=program_number)

    # Training
    for n, p in model.named_parameters():
        if p.dim() > 1 and (n != "embedding.lut.weight" and config.pretrain_emb):
            xavier_uniform_(p)

    print("TRAINABLE PARAMETERS", count_parameters(model))
    check_iter = 2000
    try:
        if config.USE_CUDA:
            model.cuda()
        model = model.train()
        best_res = 1000
        patient = 0
        writer = SummaryWriter(log_dir=config.save_path)
        weights_best = deepcopy(model.state_dict())
        data_iter = make_infinite(data_loader_tra)

        for n_iter in tqdm(range(1000000)):
            loss, ppl, bce, acc = model.train_one_batch(next(data_iter),n_iter)
            writer.add_scalars('loss', {'loss_train': loss}, n_iter)
            writer.add_scalars('ppl', {'ppl_train': ppl}, n_iter)
            writer.add_scalars('bce', {'bce_train': bce}, n_iter)
            writer.add_scalars('accuracy', {'acc_train': acc}, n_iter)
            if config.noam:
                writer.add_scalars('lr', {'learning_rate': model.optimizer._rate}, n_iter)
            if n_iter < 10000:
                continue
            if (n_iter + 1) % check_iter == 0:
                model = model.eval()
                model.epoch = n_iter
                model.__id__logger = 0
                loss_val, bleu_val, ppl_val, bce_val, acc_val = evaluate(model, data_loader_val, ty="valid", max_dec_step=50)
                writer.add_scalars('loss', {'loss_valid': loss_val}, n_iter)
                writer.add_scalars('ppl', {'ppl_valid': ppl_val}, n_iter)
                writer.add_scalars('bce', {'bce_valid': bce_val}, n_iter)
                writer.add_scalars('accuracy', {'acc_train': acc_val}, n_iter)
                model = model.train()

                if ppl_val <= best_res: 
                    best_res = ppl_val
                    patient = 0
                    weights_best = deepcopy(model.state_dict())
                    print("best_res: {}; patient: {}".format(best_res, patient))
                else:
                    patient += 1
                if patient > 2: break
                

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    ## SAVE THE BEST
    torch.save({"models": weights_best,
                'result': [loss_val, ppl_val, bce_val, acc_val], },
               os.path.join('result/model_best.tar'))
        
    print("TESTING !!!")
    model.cuda()
    model = model.eval()
    checkpoint = torch.load('result/model_best.tar', map_location=lambda storage, location: storage)
    weights_best = checkpoint['models']
    model.load_state_dict({name: weights_best[name] for name in weights_best})
    model.eval()
    loss_test, bleu_test, ppl_test, bce_test, acc_test, bsp, bsr, bsf = evaluate(model, data_loader_tst, ty="test", max_dec_step=50)

    file_summary = config.save_path + "summary.txt"
    with open(file_summary, 'w') as the_file:
        the_file.write("EVAL\tLoss\tPPL\tAccuracy\n")
        the_file.write(
            "{}\t{:.4f}\t{:.4f}\t{:.4f}".format("test", loss_test, ppl_test, acc_test))

