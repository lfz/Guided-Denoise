import sys
import numpy as np
import time
import torch
from torch.autograd import Variable
from torch import optim
from torch.nn import DataParallel

def train(epoch, net, loss_fn, data_loader, optimizer, get_lr, requires_control = True):
    start_time = time.time()
    net.eval()
    if isinstance(net, DataParallel):
        net.module.denoise.train()
    else:
        net.denoise.train()

    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    acc = []
    loss = []
    if requires_control:
        orig_acc = []
        orig_loss = []
    for i, (orig, adv, label) in enumerate(data_loader):
        adv = Variable(adv.cuda(async = True))
        label = Variable(label.cuda(async = True))
        logits = net(adv, defense = True)[-1]
        acc.append(float(torch.sum(logits.data.max(1)[1] == label.data)) / len(label))
        l = loss_fn(logits, label)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        loss.append(l.data[0])
        
        if requires_control:
            orig = Variable(orig.cuda(async = True), volatile = True)
            label = Variable(label.data, volatile = True)
            logits = net(orig)[-1]
            orig_acc.append(float(torch.sum(logits.data.max(1)[1] == label.data)) / len(label))
            l = loss_fn(logits, label)
            orig_loss.append(l.data[0])            

    acc = np.mean(acc)
    loss = np.mean(loss)
    if requires_control:
        orig_acc = np.mean(orig_acc)
        orig_loss = np.mean(orig_loss)
    end_time = time.time()
    dt = end_time - start_time

    if requires_control:
        print('Epoch %3d (lr %.5f): loss %.5f, acc %.3f, orig_loss %.5f, orig_acc %.3f, time %3.1f' % (
            epoch, lr, loss, acc, orig_loss, orig_acc, dt))
    else: 
        print('Epoch %3d (lr %.5f): loss %.5f, acc %.3f, time %3.1f' % (
            epoch, lr, loss, acc, dt))
    print


def val(epoch, net, loss_fn, data_loader, requires_control = True):
    start_time = time.time()    
    net.eval()

    acc = []
    loss = []
    if requires_control:
        orig_acc = []
        orig_loss = []
    for i, (orig, adv, label) in enumerate(data_loader):
        adv = Variable(adv.cuda(async = True), volatile = True)
        label = Variable(label.cuda(async = True), volatile = True)
        logits = net(adv, defense = True)[-1]
        acc.append(float(torch.sum(logits.data.max(1)[1] == label.data)) / len(label))
        l = loss_fn(logits, label)
        loss.append(l.data[0])
        
        if requires_control:
            orig = Variable(orig.cuda(async = True), volatile = True)
            logits = net(orig)[-1]
            orig_acc.append(float(torch.sum(logits.data.max(1)[1] == label.data)) / len(label))
            l = loss_fn(logits, label)
            orig_loss.append(l.data[0])            

    acc = np.mean(acc)
    loss = np.mean(loss)
    if requires_control:
        orig_acc = np.mean(orig_acc)
        orig_loss = np.mean(orig_loss)
    end_time = time.time()
    dt = end_time - start_time

    if requires_control:
        print('Validation: loss %.5f, acc %.3f, orig_loss %.5f, orig_acc %.3f, time %3.1f' % (
            loss, acc, orig_loss, orig_acc, dt))
    else: 
        print('Validation: loss %.5f, acc %.3f, time %3.1f' % (
            loss, acc, dt))
    print
    print

 
def test(net, data_loader, result_file_name, defense = True):
    start_time = time.time()    
    net.eval()

    acc_by_attack = {}
    for i, (adv, label, attacks) in enumerate(data_loader):
        adv = Variable(adv.cuda(async = True), volatile = True)

        adv_pred = net(adv, defense = defense)
        _, idcs = adv_pred[-1].data.cpu().max(1)
        corrects = idcs == label
        for correct, attack in zip(corrects, attacks):
            if acc_by_attack.has_key(attack):
                acc_by_attack[attack] += correct
            else:
                acc_by_attack[attack] = correct
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
