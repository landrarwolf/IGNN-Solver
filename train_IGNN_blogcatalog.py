import time
import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from utils import Evaluation
from solver_Plus.model_IGNN_solver_plus import IGNN_solver, IGNN_solver_plus
from solver_Plus.model_IGNN_ori import IGNN


import numpy
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

'''Training settings'''
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--optim', default='Adam', type=str,
                    choices=['Adam', 'SGD', 'Adagrad', 'RMSprop', 'RAdam', 'AdamW'],
                    help='optimizer to use.')  
parser.add_argument('--seed', type=int, default=2333, help='Random seed.')
parser.add_argument('--inference', action='store_true', default=False,
                    help='Inference only')

'''Parameters setting''' 
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.002,
                    help='Initial learning rate.') 
parser.add_argument('--weight_decay', type=float, default=0.005,  
                    help='Weight decay (L2 loss on parameters).')

parser.add_argument('--hidden', type=int, default=512,
                    help='Number of hidden units.') 
parser.add_argument('--threshold', type=int, default=30,
                    help='Number of hidden units.')

parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--kappa', type=float, default=0,
                    help='Projection parameter. ||W|| <= kappa/lpf(A)') 

'''Dataset/Model settings'''''
parser.add_argument('--dataset', type=str, default="blogcatalog",
                    choices=['amazon-all', 'citeseer'],
                    help='Dataset to use.')

parser.add_argument('--labelrate', type=float, default=60,  
                    help='training set fraction for amazon dataset.')

parser.add_argument('--model', type=str, default="IGNN_solver_withNN", 
                    choices=['IGNN_solver', 'IGNN_ori', 'IGNN_solver_withNN'],
                    help='model-type')

'''settings'''
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

# Load data
from config import Config
from utils2 import *

thisfolder = os.path.dirname(os.path.abspath(__file__))
initfile = os.path.join(thisfolder, 'config/', str(args.labelrate) + 'BlogCatalog.ini')
config = Config(initfile)
sadj = load_graph(args.labelrate, config, Laplace_normalization=False)
features, labels_label, idx_train, idx_test = load_data(config)  

adj = sadj  
idx_val = idx_test 
num_nodes = config.n 
num_class = config.class_num 
labels = torch.zeros(num_nodes, num_class)  

for i in range(idx_train.size(0)): 
    j = labels_label[idx_train][i]
    labels[idx_train[i], j] = 1

for i in range(idx_test.size(0)):
    j = labels_label[idx_test][i]
    labels[idx_test[i], j] = 1


def model_selector(model_name):
    models = {
        'IGNN_solver_withNN': IGNN_solver_plus(
            nfeat=features.shape[1],  
            nhid=args.hidden, 
            nclass=num_class,  
            num_node=num_nodes, 
            dropout=args.dropout, 
            kappa=args.kappa, 
        ),
        'IGNN_solver': IGNN_solver(
            nfeat=features.shape[1],  
            nhid=args.hidden, 
            nclass=num_class,  
            num_node=num_nodes,  
            dropout=args.dropout,  
            kappa=args.kappa, 
        ),
        'IGNN_ori': IGNN(
            nfeat=features.shape[1], 
            nhid=args.hidden, 
            nclass=num_class,  
            num_node=num_nodes,  
            dropout=args.dropout,
            kappa=args.kappa, 
        ),
    }
    if model_name in models:
        return models[model_name]
    else:
        raise ValueError(f"Model {model_name} does not exist")


model = model_selector(args.model)

print("Number of model parameters: ",
      sum(p.nelement() for p in model.parameters() if p.requires_grad))

optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
optimizer_hyp = (
    optim.Adam([p for p in model.parameters() if p.requires_grad],
               lr=0.008,
               weight_decay=0.005,
               )
)


# Cuda
if args.cuda is True:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


criterion = nn.BCEWithLogitsLoss()

epoch = 0
t_eval_list = []
t_total_list = []


############################################################################################
def train_f(e, threshold, simple=False):
    global epoch
    t = time.time()
    model.train()
    optimizer.zero_grad()
    model_out = model(features, adj, simple=simple, threshold=threshold)
    label_pred = model_out['label_pred']
    loss_train = criterion(label_pred[idx_train], labels[idx_train])

    loss_train.backward()
    optimizer.step()
    epoch += 1

    f1_train_micro, f1_train_macro = Evaluation(label_pred[idx_train], labels[idx_train])
    if not args.fastmode:
        model.eval()
        t_eval_begin = time.time()
        label_pred = model(features, adj, simple=simple, threshold=threshold)['label_pred']
        t_eval = time.time() - t_eval_begin
    loss_test = criterion(label_pred[idx_test], labels[idx_test])
    f1_test_micro, f1_test_macro = Evaluation(label_pred[idx_test], labels[idx_test])

    t_total = time.time() - t
    print('Epoch: {:04d}'.format(epoch),
          'Ep_f: {:04d}'.format(e + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          "f1_train_micro= {:.4f}".format(f1_train_micro),
          # "f1_train_macro= {:.4f}".format(f1_train_macro),
          # 'loss_val: {:.4f}'.format(loss_val.item()),
          # "f1_val_micro= {:.4f}".format(f1_val_micro),
          # "f1_val_micro= {:.4f}".format(f1_val_macro),
          'loss_test: {:.4f}'.format(loss_test.item()),
          # "acc_test= {:.4f}".format(acc_test),
          "f1_test_micro= {:.4f}".format(f1_test_micro),
          # "f1_test_macro= {:.4f}".format(f1_test_macro),
          'time_total: {:.4f}s'.format(t_total),
          'time_eval: {:.4f}s'.format(t_eval),
          )
    t_eval_list.append(t_eval)
    t_total_list.append(t_total)

    return f1_test_micro, f1_test_macro


################################################################################################
def train_hypsolver(e, threshold, simple=True):
    global epoch
    t = time.time()
    model.train()
    optimizer_hyp.zero_grad()
    model_out = model(features, adj, simple=simple, ep=epoch, threshold=threshold)
    loss_hypsolver = model_out['loss']

    loss_hypsolver.backward()
    optimizer_hyp.step()
    epoch += 1

    label_pred = model_out['emb']
    f1_train_micro, f1_train_macro = Evaluation(label_pred[idx_train], labels[idx_train])
    if not args.fastmode:
        model.eval()
        model_eval_out = model(features, adj, simple=simple, threshold=threshold)

        print(model_eval_out['hyp_result']['rel_trace'])
        print(model_eval_out['anderson_result']['rel_trace'])

        loss_test = model_eval_out['loss']
        label_pred = model_eval_out['emb']

        f1_test_micro, f1_test_macro = Evaluation(label_pred[idx_test], labels[idx_test])

    print('Epoch: {:04d}'.format(epoch),
          'Ep_hyp: {:04d}'.format(e + 1),
          'loss_hypsolver: {:.4f}'.format(loss_hypsolver.item()),
          "f1_train_micro= {:.4f}".format(f1_train_micro),
          'loss_test: {:.4f}'.format(loss_test.item()),
          "f1_test_micro= {:.4f}".format(f1_test_micro),
          'time: {:.4f}s'.format(time.time() - t))

    return f1_test_micro, f1_test_macro, model_eval_out


###############################################################################################
# Train model

t_total = time.time()
micro_max = 0
macro_max = 0
epoch_max = 0
micro_list = []

def solver(ep):
    global epoch
    for e in range(ep):
        train_f(e, simple=False, threshold=args.threshold)


def hypsolver(ep):
    global epoch
    for e in range(ep):
        _, _, model_eval_out = train_hypsolver(e, simple=True, threshold=args.threshold)


# Testing
def test(threshold=30):
    model.load_state_dict(torch.load(''.join(['./model_', str(args.model), '.pkl'])))
    model.eval()
    t_eval_begin = time.time()
    model_out = model(features, adj, simple=False, threshold=threshold)
    t_eval = time.time() - t_eval_begin

    label_pred = model_out['label_pred']
    # loss_test = criterion(label_pred[idx_test], labels[idx_test])
    f1_test_micro, _ = Evaluation(label_pred[idx_test], labels[idx_test])
    print("Dataset: " + args.dataset)
    print("Test set results:",
          "model= " + args.model,
          # "loss= {:.4f}".format(loss_test.item()),
          "f1_test_micro= {:.4f}".format(f1_test_micro),
          # "f1_test_macro= {:.4f}".format(f1_test_macro),
          # 'time: {:.4f}s'.format(time.time() - t_eval)
          'Inference time: {:.4f}s'.format(t_eval)
          )
    return f1_test_micro, t_eval

if args.model == "IGNN_solver" or args.model == "IGNN_ori":
    for ep in range(args.epochs):
        micro, macro = train_f(ep, simple=False, threshold=args.threshold)
        if micro >= micro_max:
            torch.save(model.state_dict(), ''.join(['./model_', str(args.model), '.pkl']))
            micro_max = micro
            macro_max = macro
            epoch_max = epoch

elif args.model == "IGNN_solver_withNN":
    # Start warming up
    print('========================== Start warming up! ==========================')
    warm_up_solver = solver(2)
    warm_up_hypsolver = hypsolver(4)
    print('Epoch_4_warm_up: {:d}'.format(epoch)) 

    # Start training model
    print('========================== Start training! ==========================')
    for ep in range(args.epochs):  
        if ep % 50 != 1:
            micro, macro = train_f(ep, simple=False, threshold=args.threshold)
            if micro >= micro_max:
                torch.save(model.state_dict(), ''.join(['./model_', str(args.model), '.pkl']))
                micro_max = micro
                macro_max = macro
                epoch_max = epoch

        else:
            print('========================== Solver training! ==========================')
            # _, _, model_eval_out = train_hypsolver(epoch, simple=True, threshold=args.threshold)
            hypsolver(ep=3)
            print('======================== Solver training END! ========================')


else:
    raise ValueError(f"Model {args.model} does not exist")


print("Optimization Finished!\n",
      'epoch:{}'.format(epoch_max),
      'micro_max: {:.4f}'.format(micro_max),
      'macro_max: {:.4f}'.format(macro_max)
      )

print("Total training time elapsed: {:.4f}s".format(time.time() - t_total))

test()
