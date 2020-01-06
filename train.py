from argparse import ArgumentParser
import os
import torch
import torch.nn as nn
import numpy as np
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import *
from facenet_pytorch import MTCNN, InceptionResnetV1

def load_dataset():
    transform = Compose([
        transforms.Resize((160,160)),
        np.float32,
        transforms.ToTensor(),
        prewhiten
    ])
    data = torchvision.datasets.ImageFolder(root='./datasets', transform=transform)
    train_size = int(len(data)* 0.8)
    valid_size = len(data) - train_size
    train_data, valid_data = torch.utils.data.random_split(data, [train_size, valid_size])
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_data, batch_size=args.val_batch_size, shuffle=True, num_workers=4)
    print("Load Data! train:{}, valid:{}".format(str(len(train_data)), str(len(valid_data))))
    return train_loader, valid_loader

def prewhiten(x):
    mean = x.mean()
    std = x.std()
    std_adj = std.clamp(min=1.0/(float(x.numel())**0.5))
    y = (x - mean) / std_adj
    return y

def setup_model(device):
    model = InceptionResnetV1(classify=True,pretrained='vggface2')
    for name, module in model._modules.items():
        module.requires_grad = False
    model.last_linear = nn.Linear(1792, 512, bias=False)
    model.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)
    model.logits = nn.Linear(512, 4) # 分類するクラス数に応じて変更する(デフォルトは4クラス)
    model.to(device)
    return model
    
def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Use Device {}".format(device))
    model = setup_model(device)
    train_loader, valid_loader = load_dataset()
    optimizer = Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epoch):
        print('Starting epoch {}/{}.'.format(epoch + 1, args.epoch))
        model.train()
        train_loss = 0
        for i_batch, data in enumerate(train_loader):
            img, label = data
            img = img.to(device).float()
            label = label.to(device).long()
            pred = model(img)
            loss = criterion(pred, label)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = train_loss / train_loader.__len__() * args.batch_size
        print('Train Loss: {}'.format(train_loss))

        model.eval()
        valid_loss = 0
        for i_batch, data in enumerate(valid_loader):
            img ,label = data
            img = img.to(device).float()
            label = label.to(device).long()
            pred = model(img)
            loss = criterion(pred, label)
            valid_loss += loss.item()
        valid_loss = valid_loss / valid_loader.__len__() * args.val_batch_size
        print('Validation Loss: {}'.format(valid_loss))

        if args.save_better_only and epoch > 1 and prev_loss < valid_loss:
            pass
        else:
            torch.save(model.state_dict(),
            os.path.join(args.out_weight_path,'epoch{}_validloss{:.4f}_trainloss{:.4f}.pth'.format(epoch + 1, valid_loss, train_loss)))
            prev_loss = valid_loss

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100,
                        help='number of epoch to train (default: 100)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--val_batch_size', type=int, default=8,
                        help='input batch size for validation (default: 8)')
    parser.add_argument("--out_weight_path", type=str, default='./weights/',
                        help="path to folder where checkpoint of trained weights will be saved")
    parser.add_argument("--save_better_only", type=bool, default=True,
                        help="save only good weights")
    args = parser.parse_args()
    required_dir = [args.out_weight_path]
    for path in required_dir:
        if not os.path.isdir(path):
            os.makedirs(path)
    run(args)