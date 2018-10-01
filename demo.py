from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
import numpy as np
import utils
import os
from os import listdir
from os.path import isfile, join
import fnmatch
from timer import Timer
import datetime
from model import centerEsti
from model import F26_N9
from model import F17_N9
from model import F35_N8

# Training settings
parser = argparse.ArgumentParser(description='parser for video prediction')
parser.add_argument('--inputFolder', type=str, required=True, help='input image folder to use')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--outputFolder', type=str, required=True, help='output image folder to use')

args = parser.parse_args()
count = 0
# load model

model1 = centerEsti()
model2 = F35_N8()
model3 = F26_N9()
model4 = F17_N9()
checkpoint = torch.load('models/center_v3.pth')
model1.load_state_dict(checkpoint['state_dict_G'])
checkpoint = torch.load('models/F35_N8.pth')
model2.load_state_dict(checkpoint['state_dict_G'])
checkpoint = torch.load('models/F26_N9_from_F35_N8.pth')
model3.load_state_dict(checkpoint['state_dict_G'])
checkpoint = torch.load('models/F17_N9_from_F26_N9_from_F35_N8.pth')
model4.load_state_dict(checkpoint['state_dict_G'])



if args.cuda:
    model1.cuda()
    model2.cuda()
    model3.cuda()
    model4.cuda()

model1.eval()
model2.eval()
model3.eval()
model4.eval()

t = Timer()
for file in listdir(args.inputFolder):
    if fnmatch.fnmatch(file, '*g'):
        count = count + 1
        if count % 1==0:
            inputFile = join(args.inputFolder, file)
            input = utils.load_image(inputFile)
            width, height= input.size
            input = input.crop((0,0, width-width%20, height-height%20))
            input.save(inputFile[:-4] + '-blurry' + inputFile[-4:])
            input_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            input = input_transform(input)
            input = input.unsqueeze(0)
            if args.cuda:
                input = input.cuda()
            input = Variable(input, volatile=True)
            t.tic()
            for kkk in range(0,1):
                output4 = model1(input)
                output3_5 = model2(input, output4)
                output2_6 = model3(input, output3_5[0], output4, output3_5[1])
                output1_7 = model4(input, output2_6[0], output3_5[0], output3_5[1], output2_6[1])
            duration = t.toc()
            print("{:.3f} s/sample.".format(duration/1))
            if args.cuda:
                output1 = output1_7[0].cpu()
                output2 = output2_6[0].cpu()
                output3 = output3_5[0].cpu()
                output4 = output4.cpu()
                output5 = output3_5[1].cpu()
                output6 = output2_6[1].cpu()
                output7 = output1_7[1].cpu()

            output_data = output1.data[0]*255
            utils.save_image(inputFile[:-4] + '-esti1' + inputFile[-4:], output_data)                
            output_data = output2.data[0]*255
            utils.save_image(inputFile[:-4] + '-esti2' + inputFile[-4:], output_data)                
            output_data = output3.data[0]*255
            utils.save_image(inputFile[:-4] + '-esti3' + inputFile[-4:], output_data)
            output_data = output4.data[0]*255
            utils.save_image(inputFile[:-4] + '-esti4' + inputFile[-4:], output_data)
            output_data = output5.data[0]*255
            utils.save_image(inputFile[:-4] + '-esti5' + inputFile[-4:], output_data)
            output_data = output6.data[0]*255
            utils.save_image(inputFile[:-4] + '-esti6' + inputFile[-4:], output_data)
            output_data = output7.data[0]*255
            utils.save_image(inputFile[:-4] + '-esti7' + inputFile[-4:], output_data)
            print('finish {:04d} processing example'.format(count))
