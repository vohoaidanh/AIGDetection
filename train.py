import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
from tensorboardX import SummaryWriter
import numpy as np
from validate import validate
from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util import Logger, get_model

import random
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


# test config
#vals = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
#multiclass = [1, 1, 1, 0, 1, 0, 0, 0] # 1 is dataset have multiclass like car, cat,  hourse...
vals = ['stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
multiclass = [1, 1, 0, 1, 0, 0, 0] # 1 is dataset have multiclass like car, cat,  hourse...

def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.classes = []
    return val_opt


if __name__ == '__main__':
    best_acc = 0
    opt = TrainOptions().parse()
    seed_torch(100)
    Testdataroot = os.path.join(opt.dataroot, 'test')
    opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
    Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'))
    print('  '.join(list(sys.argv)) )
    val_opt = get_val_opt()
    Testopt = TestOptions().parse(print_options=False)
    data_loader = create_dataloader(opt)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    
    model = Trainer(opt)
    
    def testmodel():
        print('*'*25);accs = [];aps = []
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        for v_id, val in enumerate(vals):
            Testopt.dataroot = '{}/{}'.format(Testdataroot, val)
            Testopt.classes = os.listdir(Testopt.dataroot) if multiclass[v_id] else ['']
            Testopt.no_resize = False
            Testopt.no_crop = True
            acc, ap, r_acc, f_acc, _, _ = validate(model.model, Testopt)
            accs.append(acc);aps.append(ap)
            print("({} {:12}) acc: {:.1f}; ap: {:.1f}; r_acc: {:.1f}; f_acc: {:.1f}".format(v_id, val, acc*100, ap*100, r_acc, f_acc))
        print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25) 
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    # model.eval();testmodel();
    model.train()
    print(f'cwd: {os.getcwd()}')
    early_stop_count = 0
    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(data_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), "Train loss: {} at step: {} lr {}".format(model.loss, model.total_steps, model.lr))
                train_writer.add_scalar('loss', model.loss, model.total_steps)

        if epoch % opt.delr_freq == 0 and epoch != 0:
            print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), 'changing lr at the end of epoch %d, iters %d' %
                  (epoch, model.total_steps))
            model.adjust_learning_rate()
            

        # Validation
        model.eval()
        acc, ap = validate(model.model, val_opt)[:2]
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))
        model.save_networks('last')
        if acc>=best_acc:
            print(f'acc increate {best_acc} --> {acc}, saving best model')
            best_acc = acc
            model.save_networks(f'{epoch}_best')
            early_stop_count = 0
        else:
            early_stop_count += 1
            print(f'early stop count {early_stop_count}/{opt.earlystop_epoch}')
            
        if early_stop_count == opt.earlystop_epoch:
            print(f'Early stop at epoch {epoch}')
            break
        # testmodel()
        model.train()

    #model.eval();testmodel()
    model.save_networks('last')
    
