import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import model
import config
import eval
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--lr", 
	type=float, 
	default=0.001, 
	help="learning rate")
parser.add_argument("--dropout", 
	type=float,
	default=0.0,  
	help="dropout rate")
parser.add_argument("--batch_size", 
	type=int, 
	default=256, 
	help="batch size for training")
parser.add_argument("--epochs", 
	type=int,
	default=20,  
	help="training epoches")
parser.add_argument("--top_k", 
	type=int, 
	default=10, 
	help="compute metrics@top_k")
parser.add_argument("--factors", 
	type=int,
	default=64, 
	help="predictive factors numbers in the model")
parser.add_argument("--layers", 
	type=int,
	default=4, 
	help="number of layers in MLP model")
parser.add_argument("--num_ng", 
	type=int,
	default=4, 
	help="sample negative items for training")
parser.add_argument("--test_num_ng", 
	type=int,
	default=99, 
	help="sample part of negative items for testing")
parser.add_argument("--out", 
	default=True,
	help="save model or not")
parser.add_argument("--gpu", 
	type=str,
	default="0",  
	help="gpu card ID")
args = parser.parse_args()


#prepare dataset
train_data, test_data, user_num ,item_num, train_mat = utils.load_all()

# construct the train and test datasets
train_dataset = utils.NCFData(train_data, item_num, train_mat, args.num_ng, True)
test_dataset = utils.NCFData(test_data, item_num, train_mat, 0, False)
train_loader = data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = data.DataLoader(test_dataset,batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)

#model create
if config.model == 'NeuMF-pre':
	assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
	assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
	GMF_model = torch.load(config.GMF_model_path)
	MLP_model = torch.load(config.MLP_model_path)
else:
	GMF_model = None
	MLP_model = None

#dropout 지워버릴까?
model = model.NCF(user_num,item_num,args.facors,args.layers,args.dropout,config.model,GMF_model,MLP_model)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BCELoss = nn.BCEWithLogitsLoss()

if config.model == 'NeuMF-pre':
    optimizer = optim.SGD(model.parameters(),lr=args.lr)
else:
    optimizer = optim.Adam(model.parameters(),lr=args.lr)

writer = SummaryWriter()

def train():
    count, best_hr = 0, 0
    for epoch in range(args.epochs):
        model.train() #activate dropout layer
        start = time.time()
        train_loader.dataset.ng_sample()

        for user,item,label in train_loader:
            user = user.to(device)
            item = item.to(device)
            label = label.float().to(device)

            model.zero_grad()
            score = model(user,item)
            loss = BCELoss(score,label)
            loss.backward()
            optimizer.step()

            writer.add_scalar('data/loss', loss.item(),count)
            count += 1

        model.eval() #deactivate dropout layer
        HR, NDCG, auroc = eval.metrics(model, test_loader, args.top_k)

        end = time.time() - start
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
                        time.strftime("%H: %M: %S", time.gmtime(end)))
        print("HR: {:.3f}\tNDCG: {:.3f}\tAUROC: {:.3f}".format(HR, NDCG, auroc))

        if HR > best_hr:
            best_hr, best_ndcg, best_auroc, best_epoch = HR, NDCG, auroc, epoch
            if args.out:
                if not os.path.exists(config.model_path):
                    os.mkdir(config.model_path)
                    torch.save(model, '{}{}.pth'.format(config.model_path,config.model))

    print("End. Best epoch {:03d}: HR = {:.3f}, \
            NDCG = {:.3f}, AUROC = {:.3f}".format(best_epoch, best_hr, best_ndcg, best_auroc))

if __name__ == '__main__':
    train()