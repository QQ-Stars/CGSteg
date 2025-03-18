# -*- coding: utf-8 -*-
# @Time : 2024/3/8 21:29
# @Author : blw
import os
import shutil
import time
import torch
from PIL import Image
import torchvision.utils
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class MyData(Dataset):
    # 将数据集所在的文件夹上一级传进去,作为root_dir,数据集所在的文件夹名称作为label_dir
    def __init__(self,root_dir,label_dir,transform=None):
        self.root_dir=root_dir
        self.label_dir=label_dir
        self.transform=transform  #变换
        # 将数据集所在的路径进行完整的拼接存储到path属性中
        self.path=os.path.join(self.root_dir,self.label_dir)
        # 将数据集所在的文件路径传进去,读取出来所有的文件名称
        self.img_path=os.listdir(self.path)


    # 将文件名所处的位置传进去,idx为int型作为数组的下标
    def __getitem__(self, idx):
        # 获取数据的名称
        img_name=self.img_path[idx]
        # 将数据集所在的路径与数据文件名进行拼接
        img_item_path=os.path.join(self.root_dir,self.label_dir,img_name)
        # 从文件夹内读取文件
        img=readImage(img_item_path)
        if self.transform:
            img=self.transform(img)
        # 返回出所读的数据
        return img


    def __len__(self):
        return len(self.img_path)


def readImage(img_path, channel=3):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not os.path.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        if channel == 3:
            try:
                img = Image.open(img_path).convert('RGB')
                got_img = True
            except IOError:
                print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                pass
        elif channel == 1:
            try:
                img = Image.open(img_path).convert('1')
                got_img = True
            except IOError:
                print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                pass
    return img



#图片预处理 做变换
def get_img_transforms_train(size):
    # 定义均值和标准差
    mean = [.5, .5, .5]
    std = [.5, .5, .5]
    img_transform = transforms.Compose([
        # 对图像进行随机水平翻转
        # 可以将图像从左到右进行翻转，创建一个新的图像样本
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.CenterCrop([size, size]),
        #transforms.RandomCrop([size, size],pad_if_needed=True),
        #transforms.Resize([size, size]),
        transforms.ToTensor(),
    ])
    return img_transform




def get_img_transforms_val(size):

    img_transform = transforms.Compose([
        transforms.CenterCrop([size, size]),
        #transforms.Resize([size, size]),
        transforms.ToTensor(),

    ])
    return img_transform


#图片预处理 做变换
def get_img_transforms_test(size):
    img_transform = transforms.Compose([
        transforms.CenterCrop([size, size]),
        #transforms.Resize([size, size]),
        transforms.ToTensor(),
    ])
    return img_transform


#每次实验 文件保存路径
class FileSave(object):

    def __init__(self, savepath, hostname):
        self.outckpts = savepath
        self.outlogs = savepath
        self.outcodes = savepath
        self.trainpics = savepath
        self.validationpics = savepath
        self.testPics = savepath
        self.test_Pics = savepath
        self.hostname = hostname
        self.experiment_name =''

        try:
            cur_time = time.strftime('%m-%d-%H.%M.%S', time.localtime())
            self.experiment_name = self.hostname + "_" + cur_time

            experiment_dir = "/" + self.hostname + "_" + cur_time
            if savepath == "./training" :
                self.outckpts += experiment_dir + "/checkPoints"
                self.trainpics += experiment_dir + "/train_Pics"
                self.validationpics += experiment_dir + "/validation_Pics"
                self.outlogs += experiment_dir + "/train_Logs"
                self.outcodes += experiment_dir + "/codes"
                if not os.path.exists(savepath):
                    os.makedirs(savepath)

                if not os.path.exists(self.outckpts):
                    os.makedirs(self.outckpts)
                if not os.path.exists(self.trainpics):
                    os.makedirs(self.trainpics)
                if not os.path.exists(self.validationpics):
                    os.makedirs(self.validationpics)
                if not os.path.exists(self.outlogs):
                    os.makedirs(self.outlogs)
                if not os.path.exists(self.outcodes):
                    os.makedirs(self.outcodes)

            else:
                self.testPics += experiment_dir + "/test_Pics"
                self.test_Pics = self.testPics

                self.outlogs += experiment_dir + "/train_Logs"
                if not os.path.exists(savepath):
                    os.makedirs(savepath)

                if (not os.path.exists(self.testPics)):
                    os.makedirs(self.testPics)
                if not os.path.exists(self.outlogs):
                    os.makedirs(self.outlogs)

                imgs_dirs = ['cover', 'stego', 'secret', 'revSec']
                test_imgs_dirs = [os.path.join(self.testPics, x) for x in imgs_dirs]
                for path in test_imgs_dirs:
                    os.makedirs(path)
                self.testPics = test_imgs_dirs

        except OSError:
            print("mkdir folder failed!")


#保存每次实验的代码
def save_current_codes(des_path):
    # 获取MyData.py文件路径
    MyData_file_path = os.path.realpath(__file__)
    # 获取到项目名路径
    project_path = os.path.dirname(os.path.dirname(MyData_file_path))

    train_path = os.path.join(project_path, "train1_3.py")
    new_train_path = os.path.join(des_path, "train1_3.py")
    shutil.copyfile(train_path, new_train_path)

    train_path = os.path.join(project_path, "train1_3_d.py")
    new_train_path = os.path.join(des_path, "train1_3_d.py")
    shutil.copyfile(train_path, new_train_path)

    train_path = os.path.join(project_path, "train2_3.py")
    new_train_path = os.path.join(des_path, "train2_3.py")
    shutil.copyfile(train_path, new_train_path)

    train_path = os.path.join(project_path, "train2_3_d.py")
    new_train_path = os.path.join(des_path, "train2_3_d.py")
    shutil.copyfile(train_path, new_train_path)

    test_path = os.path.join(project_path, "test.py")
    new_test_path = os.path.join(des_path, "test.py")
    shutil.copyfile(test_path, new_test_path)

    data_dir = project_path + "/data/"
    new_data_dir_path = des_path + "/data/"
    shutil.copytree(data_dir, new_data_dir_path)

    model_dir = project_path + "/models/"
    new_model_dir_path = des_path + "/models/"
    shutil.copytree(model_dir, new_model_dir_path)

    model_dir = project_path + "/gcn_lib/"
    new_model_dir_path = des_path + "/gcn_lib/"
    shutil.copytree(model_dir, new_model_dir_path)



# 输出日志信息，并保存到日志文件
def print_log(log_info, log_path, console=True):
    if console:
        print(log_info)
    # write logs into log file
    if not os.path.exists(log_path):
        fp = open(log_path, "w")
        fp.writelines(log_info + "\n")
    else:
        with open(log_path, 'a+') as f:
            f.writelines(log_info + '\n')


#  输出网络结构和参数
def print_network(net, logPath):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print_log(str(net), logPath)
    print_log('Total number of parameters: %d' % num_params, logPath)



class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



#  生成载体 stego 秘密 对比图
def save_result_pic(this_batch_size,
                    originalLabelv, ContainerImg,
                    secretLabelv, RevSecImg,
                    epoch, i, save_path, imageSize):
    originalFrames = originalLabelv.resize_(this_batch_size, 3, imageSize, imageSize)
    containerFrames = ContainerImg.resize_(this_batch_size, 3, imageSize, imageSize)
    secretFrames = secretLabelv.resize_(this_batch_size, 3, imageSize, imageSize)
    revSecFrames = RevSecImg.resize_(this_batch_size, 3, imageSize, imageSize)

    showContainer = torch.cat([originalFrames, containerFrames], 0)
    showReveal = torch.cat([secretFrames, revSecFrames], 0)
    # resultImg contains four rows: coverImg, containerImg, secretImg, RevSecImg, total this_batch_size columns
    resultImg = torch.cat([showContainer, showReveal], 0)
    resultImgName = '%s/ResultPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)
    torchvision.utils.save_image(resultImg, resultImgName, nrow=this_batch_size, padding=1, normalize=True)



def save_result_pic_test(this_batch_size,
                         originalLabelv, ContainerImg,
                         secretLabelv, RevSecImg,
                         i, save_path, imageSize):
    # For testing, save a single picture
    originalFrames = originalLabelv.resize_(this_batch_size, 3, imageSize, imageSize)
    containerFrames = ContainerImg.resize_(this_batch_size, 3, imageSize, imageSize)
    secretFrames = secretLabelv.resize_(this_batch_size, 3, imageSize, imageSize)
    revSecFrames = RevSecImg.resize_(this_batch_size, 3, imageSize, imageSize)

    originalName = '%s/cover%d.png' % (save_path[0], i)
    #originalName = '%s/%d.png' % (save_path[0], i)
    torchvision.utils.save_image(originalFrames, originalName, nrow=this_batch_size, padding=1, normalize=True)
    containerName = '%s/stego%d.png' % (save_path[1], i)
    #containerName = '%s/%d.png' % (save_path[1], i)
    torchvision.utils.save_image(containerFrames, containerName, nrow=this_batch_size, padding=1, normalize=True)
    secretName = '%s/secret%d.png' % (save_path[2], i)
    #secretName = '%s/%d.png' % (save_path[2], i)
    torchvision.utils.save_image(secretFrames, secretName, nrow=this_batch_size, padding=1, normalize=True)
    revSecName = '%s/revSec%d.png' % (save_path[3], i)
    #revSecName = '%s/%d.png' % (save_path[3], i)
    torchvision.utils.save_image(revSecFrames, revSecName, nrow=this_batch_size, padding=1, normalize=True)




def save_result_pic_test2(this_batch_size,
                         originalLabelv, ContainerImg,
                         secretLabelv, RevSecImg,
                         i, save_path, imageSize):
    # For testing, save a single picture
    originalFrames = originalLabelv.resize_(this_batch_size, 3, imageSize, imageSize)
    containerFrames = ContainerImg.resize_(this_batch_size, 3, imageSize, imageSize)
    secretFrames = secretLabelv.resize_(this_batch_size, 3, imageSize, imageSize)
    revSecFrames = RevSecImg.resize_(this_batch_size, 3, imageSize, imageSize)


    originalName = '%s/%.4d.png' % (save_path[0], i)
    torchvision.utils.save_image(originalFrames, originalName, nrow=this_batch_size, padding=1, normalize=True)

    containerName = '%s/%.4d.png' % (save_path[1], i)
    torchvision.utils.save_image(containerFrames, containerName, nrow=this_batch_size, padding=1, normalize=True)

    secretName = '%s/%.4d.png' % (save_path[2], i)
    torchvision.utils.save_image(secretFrames, secretName, nrow=this_batch_size, padding=1, normalize=True)

    revSecName = '%s/%.4d.png' % (save_path[3], i)
    torchvision.utils.save_image(revSecFrames, revSecName, nrow=this_batch_size, padding=1, normalize=True)
