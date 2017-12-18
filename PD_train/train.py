import sys
import numpy as np
import time
import torch
from torch.autograd import Variable
from torch import optim
from torch.nn import DataParallel

def train(epoch, net, data_loader, optimizer, get_lr, loss_idcs = [4], requires_control = True):
    start_time = time.time()
    net.eval()
    if isinstance(net, DataParallel):
        net.module.net.denoise.train()
    else:
        net.net.denoise.train()

    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    orig_acc = []
    acc = []
    loss = []
    if requires_control:
        control_acc = []
        control_loss = []
    for i, (orig, adv, label) in enumerate(data_loader):
        orig = Variable(orig.cuda(async = True), volatile = True)
        adv = Variable(adv.cuda(async = True), volatile = True)

        if not requires_control:
            l = net(orig, adv, requires_control = False)
        else:
            l, cl = net(orig, adv, requires_control = True)
        total_loss = l[0].mean()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        loss_values = total_loss.data.cpu().numpy()
        loss.append(loss_values)    

        if requires_control:
            loss_values = []
            loss_values.append(cl[0].mean().data.cpu().numpy()[0])
            control_loss.append(loss_values)
    loss = np.mean(loss, 0)
    if requires_control:
        control_loss = np.mean(control_loss, 0)
    end_time = time.time()
    dt = end_time - start_time
    if requires_control:
        print('Epoch %3d (lr %.5f): loss: %.3f, control loss: %.3f, time %3.1f' % (
            epoch, lr, loss, control_loss, dt))
    else: 
        print('Epoch %3d (lr %.5f): loss: %.3f, time %3.1f' % (
            epoch, lr, loss, dt))


def val(epoch, net, data_loader, requires_control = True):
    start_time = time.time()    
    net.eval()

    orig_acc = []
    acc = []
    loss = []
    if requires_control:
        control_acc = []
        control_loss = []
    for i, (orig, adv, label) in enumerate(data_loader):
        orig = Variable(orig.cuda(async = True), volatile = True)
        adv = Variable(adv.cuda(async = True), volatile = True)

        if not requires_control:
            l = net(orig, adv, requires_control = False)
        else:
            l, cl = net(orig, adv, requires_control = True)
        total_loss = l[0].mean()
        loss_values = total_loss.data.cpu().numpy()
        loss.append(loss_values)    

        if requires_control:
            loss_values = []
            loss_values.append(cl[0].mean().data.cpu().numpy()[0])
            control_loss.append(loss_values)
    loss = np.mean(loss, 0)
    if requires_control:
        control_loss = np.mean(control_loss, 0)
    end_time = time.time()
    dt = end_time - start_time
    if requires_control:
        print('Validation loss: %.3f, control loss: %.3f, time %3.1f' % (
             loss, control_loss, dt))
    else: 
        print('Validation loss: %.3f, time %3.1f' % (
             loss, dt))



    print
    print

    
def test(net, data_loader, result_file_name, defense = True):
    start_time = time.time()    
    if isinstance(net, DataParallel):
        denoiser = DataParallel(net.module.denoise)
    else:
        denoiser = net.denoise


    net.eval()
    acc_by_attack = {}
    count = {}
    for i, (clean, adv, label, attacks) in enumerate(data_loader):
        adv = Variable(adv.cuda(async = True), volatile = True)
        if defense:
            den = denoiser(adv)
        else:
            den = adv
        adv_pred = net(den, defense = False)
        _, idcs = adv_pred[-1].data.cpu().max(1)
        corrects = idcs == label
        diff = torch.abs(den.data.cpu() - clean)
        errors = diff.mean(3).mean(2).mean(1).numpy()
        for error, correct, attack in zip(errors, corrects, attacks):
            if acc_by_attack.has_key(attack):
                acc_by_attack[attack] += correct
                count[attack] += 1
                acc_by_attack[attack+'_err'] += error
            else:
                acc_by_attack[attack] = correct
                count[attack] = 1
                acc_by_attack[attack+'_err'] = error
    for attack in count.keys():
        acc_by_attack[attack+'_err'] /= float(count[attack])
    np.save(result_file_name,acc_by_attack)
    
    
class Logger(object):
    def __init__(self,logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass
