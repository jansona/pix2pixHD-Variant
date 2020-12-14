import os
from collections import OrderedDict

from numpy.lib.type_check import real
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import time
from glob import glob
import shutil
import numpy as np
import cv2
import json
from skimage.measure import compare_ssim
import random
from pytorch_fid.fid_score import calculate_fid_given_paths


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5] # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                              (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def noisy(noise_typ,row,col,ch):
    if noise_typ == "gauss":
#         row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
#         noisy = image + gauss
        noisy = gauss
        return noisy
    elif noise_typ == "random":
        random_noice = np.random.random([row, col, ch])
        return random_noice


def cal_mse(path0, path1):
    loss_func_mse = torch.nn.MSELoss(reduce=True, size_average=True)

    fs = glob("{}/*".format(path0))
    mse_loss = 0
    for f in fs:
        img_name = f.split("/")[-1]
        img_A = cv2.imread(f)
        img_B = cv2.imread("{}/{}".format(path1, img_name))
        tensor_A = torch.autograd.Variable(torch.from_numpy(img_A))
        tensor_B = torch.autograd.Variable(torch.from_numpy(img_B))
        mse_loss += loss_func_mse(tensor_A.float(), tensor_B.float())
        
    mse_avg = mse_loss / len(fs)
    return mse_avg


opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

opt.dataroot = opt.TEST_FILE_PATH
opt.checkpoints_dir = opt.MODEL_FILE
opt.results_dir = opt.OUTPUT_PATH
print("OUTPUT:", opt.OUTPUT_PATH)
opt.no_instance = True

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
if not opt.engine and not opt.onnx:
    model = create_model(opt)
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)
            
    if opt.verbose:
        print(model)
else:
    from run_engine import run_trt_engine, run_onnx
    
start_time = time.time()
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    if opt.data_type == 16:
        data['label'] = data['label'].half()
        data['inst']  = data['inst'].half()
    elif opt.data_type == 8:
        data['label'] = data['label'].uint8()
        data['inst']  = data['inst'].uint8()
    if opt.export_onnx:
        print ("Exporting to ONNX: ", opt.export_onnx)
        assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
        torch.onnx.export(model, [data['label'], data['inst']],
                          opt.export_onnx, verbose=True)
        exit(0)
    minibatch = 1 
    if opt.engine:
        generated = run_trt_engine(opt.engine, minibatch, [data['label'], data['inst']])
    elif opt.onnx:
        generated = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label'], data['inst']])
    else:        
        generated = model.inference(data['label'], data['inst'], data['image'])
        
    visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                           ('synthesized_image', util.tensor2im(generated.data[0]))])
    img_path = data['path']
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)
end_time = time.time()
print(end_time - start_time, "sec")

webpage.save()

# region 构建用于计算指标的数据
# 构建用于计算指标的数据
data4metric_path = os.path.join(web_dir, "data4metric")
if not os.path.exists(data4metric_path):
    os.mkdir(data4metric_path)

# 生成的图片
fake_imgs = glob("{}/images/*_synthesized_image.jpg".format(web_dir))
generated_path = os.path.join(data4metric_path, "generated")
if not os.path.exists(generated_path):
    os.mkdir(generated_path)
for f in fake_imgs:
    img_name = f.split("/")[-1]
    img_name = "_".join(img_name.split("_")[:2]) + ".png"
    img = cv2.imread(f)
    img_256 = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("{}/{}".format(generated_path, img_name), img_256)

# 真实图片
real_imgs = glob("{}/test_B/*".format(opt.dataroot))
ground_truth_path = os.path.join(data4metric_path, "ground_truth")
if not os.path.exists(ground_truth_path):
    os.mkdir(ground_truth_path)
for f in real_imgs:
    img_name = f.split("/")[-1]
    shutil.copy(f, "{}/{}".format(ground_truth_path, img_name))

# 另一组真实图片
another_real_imgs = glob("{}/another_ground_truth/*".format(opt.dataroot))
another_ground_truth_path = os.path.join(data4metric_path, "another_ground_truth")
if not os.path.exists(another_ground_truth_path):
    os.mkdir(another_ground_truth_path)
for f in another_real_imgs:
    img_name = f.split("/")[-1]
    shutil.copy(f, "{}/{}".format(another_ground_truth_path, img_name))

# 噪声图像
img_names = glob("{}/*.png".format(generated_path))
noise_path = os.path.join(data4metric_path, "noise")
if not os.path.exists(noise_path):
    os.mkdir(noise_path)
for f in img_names:
    img_name = f.split("/")[-1]
    noise_img = 255 * noisy("random", 256, 256, 3)
    cv2.imwrite("{}/{}".format(noise_path, img_name), noise_img)
# endregion

# region 计算指标
# SSIM
fake_imgs = glob("{}/*.png".format(generated_path))

ssims = []
for f in fake_imgs:
    img_name = f.split("/")[-1]
    
    img_A = cv2.imread(f)
    img_B = cv2.imread("{}/{}".format(ground_truth_path, img_name))
    
    ssim_single = calculate_ssim(img_A, img_B)
    
    ssims.append(ssim_single)
final_ssim = float(sum(ssims) / len(ssims))

# 分离的gt0 gt1
real_imgs = glob("{}/*.png".format(ground_truth_path))
gt0_path = os.path.join(data4metric_path, "gt0")
gt1_path = os.path.join(data4metric_path, "gt1")
os.mkdir(gt0_path)
os.mkdir(gt1_path)
random.shuffle(real_imgs)
for f in real_imgs[:int(len(real_imgs)/2)]:
    img_name = f.split("/")[-1]
    shutil.copy(f, "{}/{}".format(gt0_path, img_name))
for f in real_imgs[int(len(real_imgs)/2):]:
    img_name = f.split("/")[-1]
    shutil.copy(f, "{}/{}".format(gt1_path, img_name))

# FIDCOE
FID_generate = calculate_fid_given_paths([ground_truth_path, generated_path], 50, "cuda", 2048)
FID_noise =calculate_fid_given_paths([ground_truth_path, noise_path], 50, "cuda", 2048)
FID_gt = calculate_fid_given_paths([ground_truth_path, another_ground_truth_path], 50, "cuda", 2048)

def cal_score(FID_generate, FID_gt, FID_noise):
    return (FID_generate - FID_noise) / (FID_gt - FID_noise)

final_fidcoe = float(cal_score(FID_generate, FID_gt, FID_noise))

# MSECOE
MSE_generate = cal_mse(ground_truth_path, generated_path)
MSE_noise = cal_mse(ground_truth_path, noise_path)

final_msecoe = float(1 - MSE_generate / MSE_noise)
# endregion

print("SSIM", final_ssim)
print("FIDCOE", final_fidcoe)
print("MSECOE", final_msecoe)

# region 写报告
r_report = {"tables" : [{"tableName": "测试汇报", "结果": {
    "瓦片个数": len(dataset), 
    "SSIM": final_ssim,
    "FIDCOE": final_fidcoe,
    "MSECOE": final_msecoe,
    r"综合指标(20% SSIM, 40% FIDCOE, 40% MSECOE)": 0.2 * final_ssim + 0.4 * final_fidcoe + 0.4 * final_msecoe
}}]}

with open(opt.RESULT_PATH, "w") as jsonof:
    json.dump(r_report, jsonof, ensure_ascii=False)
# endregion
