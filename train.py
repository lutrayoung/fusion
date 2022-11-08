
import argparse
import os
import numpy as np
import math
import sys

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from config import opt
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import time
from utils_gan import (read_data,input_setup,imsave)
from loss import xLoss
#from loss.F_TargetLoss1 import F_TargetLoss
#from loss.F_BackgdLoss1 import F_BackgdLoss
import random
import cfg
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception
import models
from models.Transfusion3 import Transfusion16
from adamw import AdamW
from functions import train, validate, LinearLrDecay, load_params, copy_params, cur_stages

print(opt)

cuda = True if torch.cuda.is_available() else False


args = cfg.parse_args()
torch.cuda.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)
np.random.seed(args.random_seed)
random.seed(args.random_seed)
torch.backends.cudnn.deterministic = True


    # set tf env
#_init_inception()
#inception_path = check_or_download_inception(None)
#create_inception_graph(inception_path)

#Building model
#TFNet3=TFNet3()
#ResNet=ResNet()
#discriminator_vi = Discriminator_vi(opt.label_size)
#discriminator_ir = Discriminator_ir(opt.label_size)
Transfusion=Transfusion16 ()


if cuda:
    #TFNet3.cuda()
    #ResNet.cuda()
    #discriminator_vi.cuda()
    #discriminator_ir.cuda()
    Transfusion.cuda()




#Loading datasets
input_setup(opt,"./images/Train_ir")
input_setup(opt,"./images/Train_vi")
data_dir_ir = os.path.join('./{}'.format(opt.checkpoint_dir), "./images/Train_ir","train.h5")
data_dir_vi = os.path.join('./{}'.format(opt.checkpoint_dir), "./images/Train_vi","train.h5")
train_data_ir, train_label_ir = read_data(data_dir_ir)
train_data_vi, train_label_vi = read_data(data_dir_vi)
#data_transform = xDataTransforms.Compose([xDataTransforms.RandomCrop(opt.img_size),
                                     #xDataTransforms.ToTensor()])

#image_datasets = xDataLoader.Loader(opt=opt,
                                        #transform=data_transform,
                                        #)

#data_loaders = DataLoader(dataset=image_datasets,
                              #batch_size=opt.batch_size,
                              #shuffle=False,
                              #num_workers=opt.num_workers)

# Optimizers
if args.optimizer == "adam":
    Transfusion_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, Transfusion.parameters()),
                                     args.g_lr, (args.beta1, args.beta2))
elif args.optimizer == "adamw":
    Transfusion_optimizer = AdamW(filter(lambda p: p.requires_grad, Transfusion.parameters()),
                          args.g_lr, weight_decay=args.wd)
#gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, args.max_iter * args.n_critic)
#dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, args.max_iter * args.n_critic)
Transfusion_scheduler= LinearLrDecay(Transfusion_optimizer,args.g_lr,0.0,0,args.max_iter*args.n_critic)
  #TFNet3_optimizer = torch.optim.Adam(TFNet3.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
  #ResNet_optimizer = torch.optim.Adam(ResNet.parameters(), lr=opt.lr,betas=(opt.b1,opt.b2))
  #dis_optimizer_vi = torch.optim.Adam(discriminator_vi.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
  #dis_optimizer_ir = torch.optim.Adam(discriminator_ir.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
if cuda:
    Tensor = torch.cuda.FloatTensor
else:
    Tensor=torch.FloatTensor

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#TFNet3=nn.DataParallel(TFNet3,device_ids=[0, 1, 2,3,4])
#discriminator_ir=nn.DataParallel(discriminator_ir,device_ids=[0, 1, 2,3,4])
#discriminator_vi=nn.DataParallel(discriminator_vi,device_ids=[0, 1, 2,3,4])
#Tensor = Tensor.to(device=torch.device("cuda:0"))
Transfusion=nn.DataParallel(Transfusion,device_ids=[0,1])
#TFNet3.to(device)
#discriminator_vi.to(device)
#discriminator_ir.to(device)
Transfusion.to(device)
mse_loss = torch.nn.MSELoss()
weight_mse = [1, 10, 100, 1000]
loss_target = F_TargetLoss(['Vis', 'Inf'], kernal_size=10, device='cuda:0')
loss_backgd = F_BackgdLoss(['Vis', 'Inf'], kernal_size=10, device='cuda:0')




def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

#train
def train():

    # ----------
    #  Training
    # ----------

    # batches_done = 0
    for epoch in range(opt.max_epochs):
        print('Epoch {}/{}'.format(epoch + 1, opt.max_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        #batch_num = math.ceil(len(data_loaders.dataset) / opt.batch_size)


            
       
        Transfusion.train(True)
        batch_idxs = len(train_data_ir) // opt.batch_size
        for idx in range(0, batch_idxs):
            batch_images_ir = train_data_ir[idx * opt.batch_size: (idx + 1) * opt.batch_size]
            batch_labels_ir = train_label_ir[idx * opt.batch_size: (idx + 1) * opt.batch_size]
            batch_images_vi = train_data_vi[idx * opt.batch_size: (idx + 1) * opt.batch_size]
            batch_labels_vi = train_label_vi[idx * opt.batch_size: (idx + 1) * opt.batch_size]

            batch_images_ir = np.expand_dims(batch_images_ir, 1)
            batch_labels_ir = np.expand_dims(batch_labels_ir, 1)
            batch_images_vi = np.expand_dims(batch_images_vi, 1)
            batch_labels_vi = np.expand_dims(batch_labels_vi, 1)

            batch_images_ir = torch.autograd.Variable(torch.Tensor(batch_images_ir).cuda())
            batch_labels_ir = torch.autograd.Variable(torch.Tensor(batch_labels_ir).cuda())
            batch_images_vi = torch.autograd.Variable(torch.Tensor(batch_images_vi).cuda())
            batch_labels_vi = torch.autograd.Variable(torch.Tensor(batch_labels_vi).cuda())

            batch_images = torch.cat((batch_images_ir, batch_images_vi), 1)





        

       
            Transfusion_optimizer.zero_grad()

           
           
            fusion_img = Transfusion(batch_images)[3]
           

            
            intensity_loss=torch.mean(torch.pow(fusion_img-batch_labels_ir, 2))*opt.lam_inten_ir
           
            intensity_loss2=torch.mean(torch.pow(fusion_img-batch_labels_vi, 2))
           
            gradient_loss = xLoss.gradientLoss(fusion_img, batch_images_vi)*opt.lam_gradient
          

          
            loss = intensity_loss+intensity_loss2+gradient_loss


           
            loss.backward(retain_graph=True)
            
            Transfusion_optimizer.step()
            print_format = ['train', idx + 1, batch_idxs, loss, loss_2, intensity_loss, intensity_loss2, gradient_loss,loss_3]
            print('===> {} batch step ({}/{})\tloss:{:.5f}\tloss_2:{:.5f}\tintensity_loss:{:.5f}\tintensity_loss2:{:.5f}\tgradient_loss2:{:.5f}\tloss_3:{:.5f}'.format(*print_format))
                
            if (epoch+1) % opt.save_point == 0:
                   
                    save_path=os.path.join('./save','model%d.pkl' % (epoch +1))
                    torch.save(Transfusion,save_path)
                   
if __name__ == '__main__':
    train()
