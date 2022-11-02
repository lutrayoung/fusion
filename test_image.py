import argparse
import time
import os
import torch
import scipy
import scipy.misc
import numpy as np
# import torchvision.transforms as transform
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import DataLoader
# from data import xDataLoader
# from data import xDataTransforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
#from cvtf import *
# from NN_GAN_ATT_hy import Self_Attn
#from models.Transfusion import Transfusion9
from models.Transfusion3 import Transfusion16

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--phase', default='val', type=str, help='test phase')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
# parser.add_argument('--model_name', default='model500.pkl', type=str, help='generator model name')
# parser.add_argument("--num_workers", type=int, default=4, help="加载数据时使用子进程个数")
# parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
# parser.add_argument("--img_size", type=int, default=132, help="size of each image dimension")
# parser.add_argument("--label_size", type=int, default=120, help="size of each image dimension")
# parser.add_argument("--train_ir_dir", default='/home/l/FusionGAN_pytorch/FusionGAN_pytorch/images/Train_ir', help="训练路径")
parser.add_argument("--model_dir", default='./model_test', help="训练路径")
parser.add_argument("--test_ir_dir", default='./images/Test_ir2', help="测试路径")
parser.add_argument("--test_vi_dir", default='./images/Test_vi2', help="测试路径")
opt = parser.parse_args()

# UPSCALE_FACTOR = opt.upscale_factor



# data
def all_path(dir_path):
    file_list = []
    for maindir, subdir, file_name_list in os.walk(dir_path):
        for filename in file_name_list:
            if os.path.splitext(filename)[1] in ['.tif', '.img', '.jpg', '.png', '.bmp']:
                apath = os.path.join(maindir, filename)
                file_list.append(apath)
    return file_list

def all_path_model(dir_path):
    file_list = []
    for maindir, subdir, file_name_list in os.walk(dir_path):
        for filename in file_name_list:
            if os.path.splitext(filename)[1] in ['.pkl']:
                apath = os.path.join(maindir, filename)
                file_list.append(apath)
    return file_list

def imsave(image,path):
    return scipy.misc.imsave(path, image)

def get_img_parts(image, height, width):
    images = []
    h_cen = int(np.floor(height / 40))
    w_cen = int(np.floor(width / 40))
    for i in range(h_cen-1):
        for j in range(w_cen-1):
            img = image[:, :, i*60-20*i:(i+1)*60-20*i, j*60-20*j:(j+1)*60-20*j]
            images.append(img)
            j+=1
        i+=1
    return images, i


def recons_fusion_images(img_lists, q):
    # img_f = []
    l = len(img_lists)
    t = int(l/q)
    # h_cen = int(np.floor(height / 60))
    # w_cen = int(np.floor(width / 60))
    img_f = torch.zeros(1, 1, q * 40, t * 40).cuda()
    x = 0
    for i in range(q):
        for j in range(t):
            im = img_lists[x]
            # im1 = im[:,:,3:57,3:57]
            img_f[:, :, i * 40:(i + 1) * 40, j * 40: (j + 1) * 40] += im[:,:,10:50,10:50]
            x = x + 1
    return img_f

TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_NAME = opt.image_name
# MODEL_NAME = opt.model_name
model_list = all_path_model(opt.model_dir)
model_num = len(model_list)
model = Transfusion16().cuda().eval()

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# if cuda:
#     model = nn.DataParallel(model, device_ids=[0, 1, 2])
#
#     model.to(device)

with torch.no_grad():
    for i in range(model_num):
        model_path = model_list[i]
        model = torch.load(str(model_path))
        ir_list = all_path(opt.test_ir_dir)
        ir_list.sort()
        ir_list.sort(key=lambda x: int(x.split('/')[3][:-4]))
        vi_list = all_path(opt.test_vi_dir)
        vi_list.sort()
        vi_list.sort(key=lambda x: int(x.split('/')[3][:-4]))
        files_list = list(zip(ir_list, vi_list))
        files_num = len(files_list)
        # save_path = os.path.join(os.getcwd(), 'model'+str(i))
        # os.makedirs(save_path)
        for ii in range(0,files_num):
            #ir, vi = files_list[ii]
            ir = ir_list[ii]
            vi = vi_list[ii]
            ir = scipy.misc.imread(ir, flatten=True, mode='YCbCr').astype(np.float)
            vi = scipy.misc.imread(vi, flatten=True, mode='YCbCr').astype(np.float)
            ir = np.array(ir)
            #print (ir.shape)

            ir = (ir - 127.5) / 127.5
            vi = (np.array(vi) - 127.5) / 127.5
            #print(vi.shape)

            concat_img = np.array([ir, vi])
            #print(concat_img.shape)
            #w, h = ir.shape
            #w_1 = w % 4
            #h_1 = h % 4
            #ir = ir[0:w-w_1, 0:h-h_1]

            label = ir
            image = vi
            concat_image = concat_img
            label = torch.cuda.FloatTensor(label)
            label = torch.unsqueeze(label, 0)
            label = torch.unsqueeze(label, 0)
            #print(label.shape)
            image = torch.cuda.FloatTensor(image)
            image = torch.unsqueeze(image, 0)
            image = torch.unsqueeze(image, 0)
            concat_image = torch.cuda.FloatTensor(concat_image)
            concat_image = torch.unsqueeze(concat_image,0)
            #print(concat_image.shape)

            #w, h = vi.shape
            #w_1 = w % 4
            #h_1 = h % 4
            #vi = vi[0:w - w_1, 0:h - h_1]







            #start = time.clock()

            height = image.shape[2]
            width = image.shape[3]
            test_img_ir,nn = get_img_parts(label, height, width)
            #print(test_img_ir[0].shape)
            test_img_vi,mm = get_img_parts(image, height, width)
            test_img,pp = get_img_parts(concat_image, height, width)
            #print(test_img[0].shape)
            #k = len(test_img_vi)
            k = len(test_img_vi)

            concatimg = []
            for i in range(k):
                dd = torch.cat((test_img_ir[i],test_img_vi[i]),1)
                concatimg.append(dd)

            #print(concatimg[0].shape)



            start = time.clock()

            fusion_results = []

            for m in range(k):
                #test_img[m]=torch.cat((test_img_ir[m],test_img_vi[m]),1)
                out = model(test_img[m])[3]
                fusion_results.append(out)
                q = fusion_results[0]
            #print(fusion_results[0].shape)

            out_img = recons_fusion_images(fusion_results, nn)

            elapsed = (time.clock() - start)
            print('cost' + str(elapsed) + 's')

            # out_img = out.squeeze().cpu().detach().numpy()
            out_img = out_img.squeeze().cpu().detach().numpy()
            out_img = out_img*127.5 + 127.5
            #out_img = np.clip(out_img,0,255)

            #fusion_test_vi = test_img_vi[1]
            #fusion_test_vi = fusion_test_vi*127.5+127.5
            #fusion_test_vi = fusion_test_vi.squeeze().cpu().detach().numpy()
            #fusion_test_vi = np.clip(fusion_test_vi,0,255)

            #fusion_test_ir = test_img_ir[1]
            #fusion_test_ir = fusion_test_ir * 127.5 + 127.5
            #fusion_test_ir = fusion_test_ir.squeeze().cpu().detach().numpy()
            #fusion_test_ir = np.clip(fusion_test_ir, 0, 255)



            #imsave(out_img.astype(np.uint8), str(model_path) + str(ii+1)+".bmp")
            imsave(out_img.astype(np.uint8), "./result/"+ str(ii+1)+".bmp")

            #imsave(fusion_test_vi.astype(np.uint8),"./result/%d.bmp"% ii)
            #imsave(fusion_test_ir.astype(np.uint8), "./result/"+ str(ii)+"ir.bmp" )


