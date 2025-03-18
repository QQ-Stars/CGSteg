# -*- coding: utf-8 -*-
# @Time : 2024/7/2 18:55
# @Author : blw

import argparse
import socket
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import DataLoader
from data.MyData import print_log, save_result_pic_test, save_result_pic_test2, \
    get_img_transforms_test
from data.MyData import MyData, AverageMeter, FileSave

from models.HidingNet_15 import HidingNet
from models.RevealNet_15 import RevealNet



def main():
    ############### define global parameters ###############
    global opt, logPath, filesave, device

    # 数据集位置
    DATA_DIR = r'C:\data50000'

    parser = argparse.ArgumentParser()

    # 加载训练网络的参数权重   把default设好训练好的权重  就可 以开始测试了
    parser.add_argument('--Hnet', default='checkPoints/e811,l=0.000292,H=49.46,H=0.000159.pth',
                        help="path to Hidingnet")

    parser.add_argument('--Rnet', default='checkPoints/e811,l=0.000292,R=46.50,R=0.000133.pth',
                        help="path to Revealnet")

    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--batchSize', type=int, default=2)
    parser.add_argument('--imageSize', type=int, default=256)
    parser.add_argument('--hostname', default=socket.gethostname())

    opt = parser.parse_args()
    hostname = opt.hostname

    # 将测试结果  保存在 test 文件夹下
    savepath ="./test"
    ########################  save file  ####################
    #保存测试图片的文件的路径
    filesave = FileSave(savepath,hostname)

    # 日志路径文件名
    logPath = filesave.outlogs + '/test_%d_log.txt' % (opt.batchSize)

    ##############   get dataset   ##########################
    test_dataset = MyData(DATA_DIR, 'coco_5k', get_img_transforms_test(opt.imageSize))

    # 设置device
    try:
        # 苹果笔记本 mps加速
        use_mps = torch.backends.mps.is_available()
    except AttributeError:
        use_mps = False

    if torch.cuda.is_available():
        device = "cuda"
        print("cuda is available!")
    elif use_mps:
        device = "mps"
        print("mps is available!")
    else:
        device = "cpu"
        print("this is cpu!")

    if not opt.cuda:
        device = "cpu"
        print("this is cpu!")

    # HidingNet
    Hnet = HidingNet()
    Hnet.to(device)
    # RevealNet
    Rnet = RevealNet()
    Rnet.to(device)


    # 加载训练的模型参数
    if opt.Hnet != "":
        Hnet = torch.load(opt.Hnet,map_location=device)
        Rnet = torch.load(opt.Rnet,map_location=device)
        #Hnet.load_state_dict(torch.load(opt.Hnet))
        #Rnet.load_state_dict(torch.load(opt.Rnet))

    # MSE loss
    MSELoss = nn.MSELoss().to(device)

    test_loader = DataLoader(test_dataset, batch_size=opt.batchSize,
                             shuffle=False, num_workers=int(opt.workers))

    test(test_loader, Hnet=Hnet, Rnet=Rnet, MSELoss=MSELoss)

def test(test_loader, Hnet, Rnet, MSELoss):
    print("#################### test begin ##########################")

    start_time = time.time()
    n = 0
    Hnet.eval()
    Rnet.eval()
    Hlosses = AverageMeter()
    Rlosses = AverageMeter()

    for i, data in enumerate(test_loader, 0):
        n = i
        Hnet.zero_grad()
        Rnet.zero_grad()
        all_pics = data  # 所有 cover images and secret images
        this_batch_size = int(all_pics.size()[0] / 2)

        cover_img = all_pics[0:this_batch_size, :, :, :]
        secret_img = all_pics[this_batch_size:this_batch_size * 2, :, :, :]

        cover_img = cover_img.to(device)
        secret_img = secret_img.to(device)

        with torch.no_grad():
            container_img = Hnet(cover_img, secret_img)
            Mse_hloss = MSELoss(container_img, cover_img)
            Hlosses.update(Mse_hloss.item(), this_batch_size)

            rev_secret_img = Rnet(container_img)
            Mse_rloss = MSELoss(rev_secret_img, secret_img)
            Rlosses.update(Mse_rloss.item(), this_batch_size)

            #   保存测试图片  test   生成的图片数字编号   cover1  cover2编号   用来隐写分析
            #               test2  生成的图片数字编号   0001   0002编号      用来测  psnr ssim  lpips
            save_result_pic_test2(this_batch_size,
                                 cover_img, container_img.data,
                                 secret_img, rev_secret_img.data,
                                 i, filesave.testPics, opt.imageSize)

    test_hloss = Hlosses.avg
    test_rloss = Rlosses.avg
    test_sumloss = test_hloss + test_rloss

    test_time = time.time() - start_time
    test_log = "test_Hloss = %.6f\t test_Rloss = %.6f\t test_Sumloss = %.6f\t test-time=%.2f" % (
        test_hloss, test_rloss, test_sumloss, test_time) + "\n" + "\n"
    print_log(test_log, logPath)

    print_log("推理花费总时间:%.3f" % (test_time), logPath)
    print_log("每轮推理花费时间:%.3f" % ( test_time/(n+1) ), logPath)
    print_log("################### test end ########################", logPath)

if __name__ == '__main__':
    main()