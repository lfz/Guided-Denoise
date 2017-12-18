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
            orig_pred, adv_pred, l = net(orig, adv, requires_control = False)
        else:
            orig_pred, adv_pred, l, control_pred, cl = net(orig, adv, requires_control = True)
            
        _, idcs = orig_pred.data.cpu().max(1)
        orig_acc.append(float(torch.sum(idcs == label)) / len(label))
        _, idcs = adv_pred.data.cpu().max(1)
        acc.append(float(torch.sum(idcs == label)) / len(label))
        total_loss = 0
        for idx in loss_idcs:
            total_loss += l[idx].mean()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        loss_values = []
        for ll in l:
            loss_values.append(ll.mean().data[0])
        loss.append(loss_values)    

        if requires_control:
            _, idcs = control_pred.data.cpu().max(1)
            control_acc.append(float(torch.sum(idcs == label)) / len(label))
            loss_values = []
            for ll in cl:
                loss_values.append(ll.mean().data[0])
            control_loss.append(loss_values)
             
        #print('\torig_acc %.3f, acc %.3f, control_acc %.3f' % (
        #    orig_acc[-1], acc[-1], control_acc[-1]))
        #print('\tloss: %.5f, %.5f, %.5f, %.5f, %.5f' % (
        #    loss[-1][0], loss[-1][1], loss[-1][2], loss[-1][3], loss[-1][4]))
        #if requires_control:
        #    print('\tloss: %.5f, %.5f, %.5f, %.5f, %.5f' % (
        #        control_loss[-1][0], control_loss[-1][1], control_loss[-1][2], control_loss[-1][3], control_loss[-1][4]))
        #print

    orig_acc = np.mean(orig_acc)
    acc = np.mean(acc)
    loss = np.mean(loss, 0)
    if requires_control:
        control_acc = np.mean(control_acc)
        control_loss = np.mean(control_loss, 0)
    end_time = time.time()
    dt = end_time - start_time

    if requires_control:
        print('Epoch %3d (lr %.5f): orig_acc %.3f, acc %.3f, control_acc %.3f, time %3.1f' % (
            epoch, lr, orig_acc, acc, control_acc, dt))
    else: 
        print('Epoch %3d (lr %.5f): orig_acc %.3f, acc %.3f, time %3.1f' % (
            epoch, lr, orig_acc, acc, dt))

    print('\tloss: %.5f, %.5f, %.5f' % (
        loss[0], loss[1], loss[2]))
    
    if requires_control:
        print('\tloss: %.5f, %.5f, %.5f' % (
            control_loss[0], control_loss[1], control_loss[2]))
    print

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
            orig_pred, adv_pred, l = net(orig, adv, requires_control = False, train = False)
        else:
            orig_pred, adv_pred, l, control_pred, cl = net(orig, adv, requires_control = True, train = False)
            
        _, idcs = orig_pred.data.cpu().max(1)
        orig_acc.append(float(torch.sum(idcs == label)) / len(label))
        _, idcs = adv_pred.data.cpu().max(1)
        acc.append(float(torch.sum(idcs == label)) / len(label))
        loss_values = []
        for ll in l:
            loss_values.append(ll.mean().data[0])
        loss.append(loss_values)    

        if requires_control:
            _, idcs = control_pred.data.cpu().max(1)
            control_acc.append(float(torch.sum(idcs == label)) / len(label))
            loss_values = []
            for ll in cl:
                loss_values.append(ll.mean().data[0])
            control_loss.append(loss_values)
             
        #print('\torig_acc %.3f, acc %.3f, control_acc %.3f' % (
        #    orig_acc[-1], acc[-1], control_acc[-1]))
        #print('\tloss: %.5f, %.5f, %.5f, %.5f, %.5f' % (
        #    loss[-1][0], loss[-1][1], loss[-1][2], loss[-1][3], loss[-1][4]))
        #if requires_control:
        #    print('\tloss: %.5f, %.5f, %.5f, %.5f, %.5f' % (
        #        control_loss[-1][0], control_loss[-1][1], control_loss[-1][2], control_loss[-1][3], control_loss[-1][4]))
        #print 
    
    orig_acc = np.mean(orig_acc)
    acc = np.mean(acc)
    loss = np.mean(loss, 0)
    if requires_control:
        control_acc = np.mean(control_acc)
        control_loss = np.mean(control_loss, 0)
    end_time = time.time()
    dt = end_time - start_time

    if requires_control:
        print('Validation: orig_acc %.3f, acc %.3f, control_acc %.3f, time %3.1f' % (
            orig_acc, acc, control_acc, dt))
    else:
        print('Validation: orig_acc %.3f, acc %.3f, time %3.1f' % (
            orig_acc, acc, dt))

    print('\tloss: %.5f, %.5f, %.5f' % (
        loss[0], loss[1], loss[2]))
    
    if requires_control:
        print('\tloss: %.5f, %.5f, %.5f' % (
            control_loss[0], control_loss[1], control_loss[2]))
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
    print(result_file_name)
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
