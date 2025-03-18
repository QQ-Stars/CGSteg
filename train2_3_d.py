# -*- coding: utf-8 -*-
# @Time : 2024/7/6 11:14
# @Author : blw
import argparse
import socket
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from data.MyData import  print_log, print_network, save_current_codes, \
    get_img_transforms_train, get_img_transforms_val
from data.MyData import MyData, AverageMeter, FileSave
from models.HidingNet_15 import HidingNet as HidingNet
from models.RevealNet_15 import RevealNet as RevealNet
from models.MBSSIM import MBSSIM
import lpips

import numpy as np
from skimage.metrics import peak_signal_noise_ratio,structural_similarity

from models.discriminator import XuNet


# 使用三个损失训练，训练图片加载两次
# 加入鉴别器， 进行对抗训练


# 使用 skimage 计算 PSNR
def calculate_psnr_skimage(img1, img2):
    """
        calculate psnr in Y channel.
    """
    img_1 = (np.array(img1).astype(np.float64)*255).astype(np.float64)
    img_2 = (np.array(img2).astype(np.float64)*255).astype(np.float64)
    img1_y = rgb2ycbcr(img_1.transpose(0,2,3,1))
    img2_y = rgb2ycbcr(img_2.transpose(0,2,3,1))
    return peak_signal_noise_ratio(img1_y, img2_y,data_range=255)


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


# train2-3.py  Image loading 2  ,  3 loss functions
def main():
    ############### define global parameters ###############
    global opt, writer, logPath, filesave, optimizerH, optimizerR,iters_per_epoch,\
        device, MBSSIM_Loss, loss_fn_vgg, Dnet, optimizer_discriminator

    # 设置随机种子
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN自动寻求最佳卷积算法优化
    cudnn.benchmark = True

    # 数据集位置
    DATA_DIR = 'C:\data1000'  #C:\ImageNet

    parser = argparse.ArgumentParser()
    parser.add_argument('--Hnet', default='',
                        help="path to Hidingnet (to continue training)")
    parser.add_argument('--Rnet', default='',
                        help="path to Revealnet (to continue training)")

    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--batchSize', type=int, default=32)
    parser.add_argument('--imageSize', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--beta_1', type=float, default=1)
    parser.add_argument('--beta_2', type=float, default=1)
    parser.add_argument('--beta_3', type=float, default=5)
    parser.add_argument('--beta_4', type=float, default=5)
    parser.add_argument('--beta_5', type=float, default=1)
    parser.add_argument('--beta_6', type=float, default=1)
    parser.add_argument('--beta_a', type=float, default=0.9)
    parser.add_argument('--hostname', default=socket.gethostname(), )
    parser.add_argument('--logFrequency', type=int, default=500)
    parser.add_argument('--save_weight_begin', type=int, default=1)  # 多少轮后，开始保存权重文件

    parser.add_argument('--resultPicFrequency', type=int, default=50)
    parser.add_argument('--epochSave', type=int, default=100,
                        help='100个epoch保存一次 stego')

    #################  output configuration   ###############
    opt = parser.parse_args()
    hostname = opt.hostname
    savepath ="./training"

    ########################  save file  ####################
    #保存权重，日志，训练的文件的路径
    filesave = FileSave(savepath,hostname)
    # 日志路径文件名
    logPath = filesave.outlogs + '/train2-3_%d_log.txt' % (opt.batchSize)
    print_log("#### train2-3.py  Image loading 2 ,  3 loss functions #####", logPath)
    # 新建日志文件，并将opt参数保存到日志文件
    print_log(str(opt), logPath)
    # 保存代码文件
    save_current_codes(filesave.outcodes)

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

    # tensorboardX writer
    writer = SummaryWriter("training/"+filesave.experiment_name)
    ##############   get dataset   ############################
    train_cover_dataset = MyData(DATA_DIR, 'train', get_img_transforms_train(opt.imageSize))
    train_secret_dataset = MyData(DATA_DIR, 'train', get_img_transforms_train(opt.imageSize))

    val_dataset = MyData(DATA_DIR,'val',get_img_transforms_val(opt.imageSize))
    train_cover_loader = DataLoader(train_cover_dataset, batch_size=opt.batchSize,
                                    shuffle=True, num_workers=int(opt.workers))
    train_secret_loader = DataLoader(train_secret_dataset, batch_size=opt.batchSize,
                                     shuffle=True, num_workers=int(opt.workers))
    val_loader = DataLoader(val_dataset, batch_size=opt.batchSize*2,
                            shuffle=False, num_workers=int(opt.workers))

    # 每轮迭代的次数
    iters_per_epoch = len(train_cover_loader)
    # HidingNet
    Hnet = HidingNet().to(device)
    # RevealNet
    Rnet = RevealNet().to(device)
    # discriminator
    Dnet = XuNet().to(device)

    # 加载训练的模型参数
    if opt.Hnet != "":
        Hnet = torch.load(opt.Hnet, map_location=device)
        Rnet = torch.load(opt.Rnet, map_location=device)
        #Hnet.load_state_dict(torch.load(opt.Hnet))
        #Rnet.load_state_dict(torch.load(opt.Rnet))

    # setup optimizer
    optimizerH = optim.Adam(Hnet.parameters(), lr=opt.lr, betas=(opt.beta_a, 0.999))
    schedulerH = ReduceLROnPlateau(optimizerH, mode='min', factor=0.5, patience=10, verbose=True)

    optimizerR = optim.Adam(Rnet.parameters(), lr=opt.lr, betas=(opt.beta_a, 0.999))
    schedulerR = ReduceLROnPlateau(optimizerR, mode='min', factor=0.5, patience=10, verbose=True)

    optimizer_discriminator = optim.Adam(Dnet.parameters(), lr=1e-4, weight_decay=0)

    # MSE loss
    MSE_Loss = nn.MSELoss().to(device)
    # MSE_Loss = nn.L1Loss().to(device)
    MBSSIM_Loss = MBSSIM(kernel_size=4, stride=4, repeat_time=5, patch_height=opt.imageSize, patch_width=opt.imageSize).to(device)
    loss_fn_vgg = lpips.LPIPS(net='vgg', spatial=True, lpips=True).to(device)

    # 计算一下训练总时间
    start_ = time.time()
    min_mse = 1000
    max_sum_psnr = 0
    for epoch in range(1,opt.epochs+1):
        print_log("#################### train begin #######################", logPath)
        train_loader = zip(train_cover_loader, train_secret_loader)
        train(train_loader, epoch, Hnet=Hnet, Rnet=Rnet, MSE_Loss=MSE_Loss)

        print_log("################### validation begin ###################", logPath)
        mse_hloss, mse_rloss,val_mse_loss, PSNR_C_S, PSNR_S_R = validation(val_loader, epoch, Hnet=Hnet, Rnet=Rnet, MSE_Loss=MSE_Loss)
        sum_psnr = PSNR_C_S + PSNR_S_R
        ####################### adjust learning rate ############################
        schedulerH.step(mse_hloss)
        schedulerR.step(mse_rloss)

        if val_mse_loss < min_mse or sum_psnr > max_sum_psnr:
            if val_mse_loss < min_mse:
                min_mse = val_mse_loss
            if sum_psnr > max_sum_psnr:
                max_sum_psnr = sum_psnr
            if epoch > opt.save_weight_begin:
                # 保存模型参数
                torch.save(Hnet,
                           '%s/e%d,l=%.6f,H=%.2f,H=%.6f.pth' % (
                           filesave.outckpts, epoch, mse_hloss + mse_rloss, PSNR_C_S, mse_hloss))
                torch.save(Rnet,
                           '%s/e%d,l=%.6f,R=%.2f,R=%.6f.pth' % (
                           filesave.outckpts, epoch, mse_hloss + mse_rloss, PSNR_S_R, mse_rloss))


    end_ = time.time()
    print_log('训练总时间：%.2fh' % ((end_ - start_)/3600), logPath)
    writer.close()


def train(train_loader, epoch, Hnet, Rnet, MSE_Loss):
    batch_time = AverageMeter()

    MSE_C_S = AverageMeter()
    MSE_S_R = AverageMeter()

    SSIM_C_S = AverageMeter()
    SSIM_S_R = AverageMeter()

    LPIPS_C_S = AverageMeter()
    LPIPS_R_S = AverageMeter()

    PSNR_C_S = AverageMeter()
    PSNR_S_R = AverageMeter()

    Adv_stego_loss = AverageMeter()
    D_cover = AverageMeter()
    D_stego = AverageMeter()
    D_loss = AverageMeter()

    Sum_Loss = AverageMeter()

    # switch to train mode
    Hnet.train()
    Rnet.train()
    Dnet.train()


    start_time = time.time()
    for i, (cover_img, secret_img) in enumerate(train_loader, 0):

        this_batch_size = int(cover_img.size()[0])  # get true batch size of this step
        cover_img = cover_img.to(device)
        secret_img = secret_img.to(device)

        ############## Train the (encoder-decoder) ##############
        Hnet.zero_grad()
        Rnet.zero_grad()
        for p in Dnet.parameters():
            p.requires_grad = False
        container_img = Hnet(cover_img, secret_img)
        mse_h_loss = MSE_Loss(container_img, cover_img)
        ssim_c_s=MBSSIM_Loss(container_img, cover_img)
        h_loss_vgg = loss_fn_vgg(container_img, cover_img)

        MSE_C_S.update(mse_h_loss.item(), 1)
        SSIM_C_S.update(ssim_c_s.item(), 1)
        LPIPS_C_S.update(h_loss_vgg.mean().item(),1)

        # with torch.no_grad():
        #     psnr_c_s = psnr_between_batches(container_img, cover_img)
        #     PSNR_C_S.update(psnr_c_s,1)

        rev_secret_img = Rnet(container_img)
        mse_r_loss = MSE_Loss(rev_secret_img, secret_img)
        ssim_s_r = MBSSIM_Loss(rev_secret_img, secret_img)
        r_loss_vgg = loss_fn_vgg(rev_secret_img, secret_img)

        MSE_S_R.update(mse_r_loss.item(), 1)
        SSIM_S_R.update(ssim_s_r.item(), 1)
        LPIPS_R_S.update(r_loss_vgg.mean().item(), 1)

        adv_stego = Dnet(container_img)
        adv_stego_loss = adv_stego.mean()
        Adv_stego_loss.update(adv_stego_loss.mean().item(), 1)

        # with torch.no_grad():
        #     psnr_s_r = psnr_between_batches(rev_secret_img, secret_img)
        #     PSNR_S_R.update(psnr_s_r,1)

        SumLoss_mse_ssim = opt.beta_1 * mse_h_loss + opt.beta_2 * mse_r_loss + \
                           ssim_c_s * opt.beta_3 + ssim_s_r * opt.beta_4  + \
                           h_loss_vgg.mean() * opt.beta_5 + r_loss_vgg.mean() *opt.beta_6 + adv_stego_loss
        Sum_Loss.update(SumLoss_mse_ssim.item(), 1)
        SumLoss_mse_ssim.backward()
        optimizerH.step()
        optimizerR.step()

        ############## Train the discriminator ##############
        Diters = 2
        j = 0
        while j < Diters :
            for p in Dnet.parameters():
                p.requires_grad = True
            Dnet.zero_grad()
            d_cover = Dnet(cover_img)
            d_stego = Dnet(container_img.detach())
            d_loss = d_cover.mean() - d_stego.mean()
            D_cover.update(d_cover.mean().item(),1)
            D_stego.update(d_stego.mean().item(), 1)
            D_loss.update(d_loss.mean().item(), 1)

            d_loss.backward()
            optimizer_discriminator.step()

            for p in Dnet.parameters():
                p.data.clamp_(-0.05, 0.05)
            j = j + 1


        log = '[%d/%d][%d/%d]\tMSE_C_S: %.6f\t MSE_S_R: %.6f\t MSE_sum: %.6f \t adv_steggo_loss: %.6f' % (
            epoch, opt.epochs, i, iters_per_epoch, MSE_C_S.val, MSE_S_R.val, MSE_C_S.val+MSE_S_R.val, adv_stego_loss) + '\n'
        log = log + 'd_cover: %.6f\t d_stego: %.6f \t d_loss: %.6f' % (D_cover.val, D_stego.val,D_loss.val)

        if i % opt.logFrequency == 0:
            print_log(log, logPath)

        # 生成载体 stego 秘密 对比图
        # if epoch % opt.epochSave == 0 and i % opt.resultPicFrequency == 0:
        #     save_result_pic(this_batch_size,
        #                     cover_img, container_img.data,
        #                     secret_img, rev_secret_img.data,
        #                     epoch, i, filesave.trainpics, opt.imageSize)

    batch_time.update(time.time() - start_time)
    # epcoh log
    epoch_log = "one epoch time is %.2fs=================================" % (batch_time.sum) + "\n"
    epoch_log = epoch_log + "epoch learning rate: optimizerH_lr = %.8f      optimizerR_lr = %.8f" % (
        optimizerH.param_groups[0]['lr'], optimizerR.param_groups[0]['lr']) + "\n"
    epoch_log = epoch_log + "MSE_C_S=%.6f\tMSE_S_R=%.6f\tMSE_sum=%.6f" % (
        MSE_C_S.avg, MSE_S_R.avg, MSE_C_S.avg+MSE_S_R.avg)+ "\n"
    epoch_log = epoch_log + "SSIM_C_S=%.6f\tSSIM_S_R=%.6f\tSSIM_sum=%.6f" % (
        SSIM_C_S.avg, SSIM_S_R.avg,SSIM_C_S.avg+SSIM_S_R.avg) + "\n"
    epoch_log = epoch_log + "VGG_C_S=%.6f\tVGG_S_R=%.6f\tVGG_sum=%.6f" % (
        LPIPS_C_S.avg, LPIPS_R_S.avg, LPIPS_C_S.avg + LPIPS_R_S.avg) + "\n"
    epoch_log = epoch_log + "PSNR_C_S=%.3f\t\tPSNR_S_R=%.3f" % (
        PSNR_C_S.avg, PSNR_S_R.avg) + "\n"
    epoch_log = epoch_log + "Adv_stego_loss=%.6f" % (
        Adv_stego_loss.avg) + "\n"
    epoch_log = epoch_log + 'd_cover: %.6f\t d_stego: %.6f \t d_loss: %.6f' % (D_cover.avg, D_stego.avg, D_loss.avg)+ "\n"
    print_log(epoch_log, logPath)

    # record lr
    writer.add_scalar("lr/H_lr", optimizerH.param_groups[0]['lr'], epoch)
    writer.add_scalar("lr/R_lr", optimizerR.param_groups[0]['lr'], epoch)
    writer.add_scalar("lr/beta_a", opt.beta_a, epoch)
    # record loss
    writer.add_scalar('train/H_MSE', MSE_C_S.avg, epoch)
    writer.add_scalar('train/R_MSE', MSE_S_R.avg, epoch)
    writer.add_scalar('train/Sum_loss', Sum_Loss.avg, epoch)


def validation(val_loader, epoch, Hnet, Rnet, MSE_Loss):
    start_time = time.time()
    Hnet.eval()
    Rnet.eval()

    MSE_C_S = AverageMeter()
    MSE_S_R = AverageMeter()

    PSNR_C_S = AverageMeter()
    PSNR_S_R = AverageMeter()

    for i, data in enumerate(val_loader, 0):
        Hnet.zero_grad()
        Rnet.zero_grad()

        all_pics = data
        this_batch_size = int(all_pics.size()[0] / 2)
        cover_img = all_pics[0:this_batch_size, :, :, :]
        secret_img = all_pics[this_batch_size:this_batch_size * 2, :, :, :]

        cover_img = cover_img.to(device)
        secret_img = secret_img.to(device)

        with torch.no_grad():
            container_img = Hnet(cover_img, secret_img)
            rev_secret_img = Rnet(container_img)
            mse_h_loss = MSE_Loss(container_img, cover_img)
            mse_r_loss = MSE_Loss(rev_secret_img, secret_img)
            MSE_C_S.update(mse_h_loss.item(), 1)
            MSE_S_R.update(mse_r_loss.item(), 1)

            # 计算各种指标
            # 拷贝进内存以方便计算
            cover_img = cover_img.cpu()
            secret_img = secret_img.cpu()
            container_img = container_img.cpu()
            rev_secret_img =  rev_secret_img.cpu()

            # 计算 Y 通道 PSNR
            psnr_c_temp = calculate_psnr_skimage(cover_img, container_img)
            psnr_s_temp = calculate_psnr_skimage(secret_img, rev_secret_img)
            PSNR_C_S.update(psnr_c_temp, 1)
            PSNR_S_R.update(psnr_s_temp, 1)

    mse_hloss = MSE_C_S.avg
    mse_rloss = MSE_S_R.avg
    val_mse_loss = mse_hloss * opt.beta_1 + mse_rloss * opt.beta_2

    val_time = time.time() - start_time
    val_log = "validation[%d] MSE_C_S=%.6f\t  MSE_S_R=%.6f\t  MSE_sum=%.6f\t  validation time=%.2fs" % (
        epoch, mse_hloss, mse_rloss, mse_hloss+mse_rloss, val_time)+ "\n"
    val_log = val_log + "val_psnr_h=%.3f\t  val_psnr_r=%.3f" % (
        PSNR_C_S.avg, PSNR_S_R.avg) + "\n"
    print_log(val_log, logPath)

    writer.add_scalar('validation/H_MSE', mse_hloss, epoch)
    writer.add_scalar('validation/R_MSE', mse_rloss, epoch)
    writer.add_scalar('validation/Sum_loss', val_mse_loss, epoch)
    return mse_hloss, mse_rloss, val_mse_loss, PSNR_C_S.avg, PSNR_S_R.avg

def psnr1(image1, image2):
    """each element should be in [0, 1]"""
    mse = torch.mean((image1 - image2) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1.0
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

def psnr_between_batches(batch1, batch2):
    batch_size = batch1.size(0)
    average = 0.
    for sample_id in range(batch_size):
        p = psnr1(batch1[sample_id], batch2[sample_id])
        average += p
    average /= batch_size
    return average


if __name__ == '__main__':
    main()

