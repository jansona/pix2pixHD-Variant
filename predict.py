import os
from collections import OrderedDict
from torch.autograd import Variable
from options.predict_options import PredictOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import cv2
import time
import re
import numpy as np
from glob import glob

# 地图尺度与坐标位宽的映射
zoom2width = {
    15: 5,
    17: 6,
    18: 6
}

def integrate_tiles(d_name, tile_mat):

    for line in tile_mat:
        for tile in line:
            if not os.path.exists("{}/{}".format(d_name, tile)):
                print(d_name, tile)
    
    def assemble_row(row_files):
        
        tile_cated = cv2.imread(os.path.join(d_name, row_files[0]))
        
        for file in row_files[1:]:
            temp_tile = cv2.imread(os.path.join(d_name, file))
            array_temp = np.array(temp_tile)
            if array_temp.ndim == 0:
                break
            tile_cated = np.concatenate((tile_cated, temp_tile), axis=1)
            
        return tile_cated
    
    rows = []
    
    for row in tile_mat:
        rows.append(assemble_row(row))
        
    map_cated = rows[0]
    
    for row in rows[1:]:
        map_cated = np.concatenate((map_cated, row), axis=0)
        
    return map_cated

def statis_value(in_path, suffix):
    name_list = os.listdir(in_path)
    name_list = list(filter(lambda x: re.match("\d+_\d+.".format(suffix), x), name_list))
    y_list = []
    x_list = []
    for name in name_list:
        name = name.split("/")[-1]
        name = name.split(".")[0]
        a = name.split('_',2)
        y_list.append(int(a[0]))
        x_list.append(int(a[1]))
    x_min,x_max = min(x_list),max(x_list)
    y_min,y_max = min(y_list),max(y_list)
    return x_min,x_max,y_min,y_max

opt = PredictOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt, "plain")
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
        
    visuals = OrderedDict([('', util.tensor2im(generated.data[0]))])
    img_path = data['path']
    print('process image... %s' % img_path)
    visualizer.save_images_predict(webpage, visuals, img_path)
end_time = time.time()
print(end_time - start_time, "seconds")

webpage.save()

# automatic integrate and autocontrast - for platform
print("start integrating...")
starttime = time.time()

# in_path = webpage.get_image_dir()
in_path = web_dir
# out_path = in_path[:-6] + "integrated"
out_path = in_path

temp_suffix_name = glob("{}/*".format(in_path))[0].split('.')[-1]

x_min, x_max, y_min, y_max = statis_value(in_path, temp_suffix_name)
x_size = x_max - x_min + 1
y_size = y_max - y_min + 1
zoom = opt.zoom
coord_width = zoom2width[zoom]
        
base_path = in_path + "/"
file_template = "{:0%dd}_{:0%dd}." %(coord_width, coord_width) + temp_suffix_name
tile_files = []
for i in range(x_size):
    temp_list = []
    for j in range(y_size):
        temp_list.append(file_template.format(y_min + j, x_min + i))
        
    tile_files.append(temp_list)

map_pic = integrate_tiles(web_dir, tile_files)
cv2.imwrite(opt.RESULT_PATH, map_pic)

# for root, dirs, files in os.walk(web_dir, topdown=False):
#     for name in files:
#         os.remove(os.path.join(root, name))
#     for name in dirs:
#         os.rmdir(os.path.join(root, name))
# os.removedirs(web_dir)

lasttime = time.time()
print('Integration done!', 'Total Time Cost: ', lasttime - starttime, 'seconds')
