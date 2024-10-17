import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import re

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models import vgg19
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal as Norm
import cv2

from PIL import Image
import torchvision.transforms as transforms


def main():

    technique = '_tmfi' # DA or Base Model (HD)?
    perf_metric = 'NSS'
    separated_directories = True # there are cases where saliency directory contains two subdirectories: frames (sal. prediction) and images (sal. prediction overlayed on the original video frame)

    # set paths to saliency maps and fixation maps
    #path_saliency_maps = 
    path_saliency_maps = ''
    
    #path_fixation_maps = 
    path_fixation_maps = ''
    
    path_results = ''
    if not os.path.isdir(path_results):
        os.makedirs(path_results)
    path_extra_info = path_results + ''

    # retrieve directories with the several folders that contain the video frames
    list_saliency_maps = [d for d in os.listdir(path_saliency_maps) if os.path.isdir(os.path.join(path_saliency_maps, d))]
    list_saliency_maps.sort()
    list_fixation_maps = [d for d in os.listdir(path_fixation_maps) if os.path.isdir(os.path.join(path_fixation_maps, d))]
    list_fixation_maps.sort()

    # to upload images with library PIL
    transform = transforms.Compose([
    transforms.Grayscale(),  # Convert to grayscale
    transforms.ToTensor(),   # Convert to PyTorch Tensor
    ])


    # loop through each folder and save results as a list
    results_list = []
    video_name_list = [] # for collecting detailed info (info per frame)
    frame_number_list = [] # for collecting detailed info (info per frame)
    metric_all_videos_list = [] # for collecting detailed info (info per frame)
    for dname in list_fixation_maps:

        video_name = dname

        if separated_directories:
            # path to video frames
            path_predicted = os.path.join(path_saliency_maps, video_name)  # prediction/saliency maps
            path_gt = os.path.join(path_fixation_maps, dname)  # for eduVideos, DIEM dataset
        else:
            # path to video frames
            path_predicted = os.path.join(path_saliency_maps, video_name)  # prediction/saliency maps
            path_gt = os.path.join(path_fixation_maps, dname)  # ground truth/fixations maps

        # list of frames ground truth
        frames_gt_list = [f for f in listdir(path_gt) if isfile(join(path_gt, f))]
        frames_gt_list = [val for val in frames_gt_list if (val.endswith(".png") or val.endswith(".jpg"))]
        frames_gt_list.sort()  # sort

        # list of frames predicted
        frames_predicted_list = [f for f in listdir(path_predicted) if isfile(join(path_predicted, f))]
        frames_predicted_list = [val for val in frames_predicted_list if (val.endswith(".png") or val.endswith(".jpg"))]
        frames_predicted_list.sort()  # sort


        # just for registration: detect when ground truth frames are less than predicted frames (because there was not enough data to create the gt/fixation maps)
        if len(frames_gt_list) < len(frames_predicted_list):
            # save information
            with open(path_extra_info, 'a') as file:
                print('video:', dname, 'number of gt/fixation maps less than predicted frames. ~Processed frames:', len(frames_gt_list), file=file)
        elif len(frames_gt_list) == len(frames_predicted_list):
            with open(path_extra_info, 'a') as file:
                print('video:', dname, 'number of gt/fixation maps same as predicted frames . ~Processed frames:', len(frames_gt_list), file=file)
        else: # if	len(frames_gt_list) > len(frames_predicted_list)
            with open(path_extra_info, 'a') as file:
                print('video:', dname, 'number of gt/fixation maps more than predicted frames. ~Processed frames:', len(frames_predicted_list), file=file)


        # compute metric
        metric_list = []
        for i in range(len(frames_gt_list)):

            # sometimes # of ground truth (gt/fixation maps) are more than saliency/fixation maps (because participants were longer in the computer), so, detect this and finish the loop
            if i + 1 > len(frames_predicted_list):
                break

            # make sure that that background frame has corresponding salience/fixation map (sometimes there is no corresponding
            # fixation map because there was not enough data to create the fixation/density plot. Compare exluding the extensions, e.g., 'jpg and png
            #frame_number = str(int(frames_gt_list[i].split('.')[0]))
            frame_number_reg_exp_gt = re.match(r'.*?(\d+).*', frames_gt_list[i])  # regular expression
            frame_number_gt = str(int(frame_number_reg_exp_gt.group(1)))  # extract reg exp and make it integer
            #if any(frame_number == str(int(file.split('.')[0])) for file in frames_predicted_list):
            if any(frame_number_gt == str(int(re.match(r'.*?(\d+).*', file).group(1))) for file in frames_predicted_list):
                # upload images
                #image_gt = cv2.imread(path_gt + '/' + frames_gt_list[i], cv2.IMREAD_GRAYSCALE)  # image ground truth
                path_gt_image = os.path.join(path_gt, frames_gt_list[i]) # better load with Image from PIL to allow later compute with cuda
                image_gt = Image.open(path_gt_image).convert('L')  
                image_gt = transform(image_gt)
                #image_predicted = cv2.imread(path_predicted + '/' + frames_predicted_list[i], cv2.IMREAD_GRAYSCALE)
                path_predicted_image = os.path.join(path_predicted, frames_predicted_list[i])
                image_predicted = Image.open(path_predicted_image).convert('L')
                image_predicted = transform(image_predicted)

                '''
                # resize if needed:
                if image_gt.shape != image_predicted.shape:
                    if image_gt.shape[0] * image_gt.shape[1] < image_predicted.shape[0] * image_predicted.shape[1]:
                        new_shape = (image_gt.shape[1], image_gt.shape[0])
                        image_predicted = cv2.resize(image_predicted, new_shape)
                    else:
                        new_shape = (image_predicted.shape[1], image_predicted.shape[0])
                        image_gt = cv2.resize(image_gt, new_shape)
                '''

                # Resize if needed
                if image_gt.size() != image_predicted.size():
                    # Resize image_predicted to match image_gt size
                    image_predicted = transforms.functional.resize(image_predicted, image_gt.shape[1:])


                # calculate matric
                #metric = NSS(image_predicted, image_gt)
                metric = NSS(image_gt, image_predicted)
                metric = metric.item() # the result from above is in format tensor(2.6545), so I have to extrac the value with .item
                metric_list.append(metric)  # metric is a tensor, so to get the value inside we use item()
                print('NSS is: ', metric, 'of frame #: ', i + 1, ', experiment & video: ', dname)

                # store individual results in a list to later make a table and save as .csv
                video_name_list.append(dname)
                frame_number_list.append(i+1)
                metric_all_videos_list.append(metric)

        # get statistics
        metric_series = pd.Series(metric_list)
        results = pd.DataFrame({dname:metric_series.describe()}) # dname is the name of the column
        results_list.append(results)

    # results_list is a list of dataframes that have the same index column(s)
    results_df = pd.concat(results_list, axis=1)

    # save the resulting dataframe with summary statistics as a CSV file
    results_df.to_csv(path_results + '/' + perf_metric + technique + '_new.csv', index=True)

    # save detailed results (metric per video frame per video) as a CSV file
    detailed_results_df = pd.DataFrame({'video_name': video_name_list, 'frame_number': frame_number_list, 'value_CC': metric_all_videos_list})
    detailed_results_df.to_csv(path_results + '/' + 'detailed_' + perf_metric + technique + '.csv', index=True)


    print('RESULTS SAVED!')



#def nss(s_map, gt):
def NSS(gt, s_map):
    
    if s_map.size() != gt.size():
        s_map = s_map.cpu().squeeze(0).numpy()
        s_map = torch.FloatTensor(cv2.resize(s_map, (gt.size(2), gt.size(1)))).unsqueeze(0)
        s_map = s_map.cuda()
        gt = gt.cuda()
    assert s_map.size() == gt.size()
    batch_size = s_map.size(0)
    w = s_map.size(1)
    h = s_map.size(2)
    mean_s_map = torch.mean(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)
    std_s_map = torch.std(s_map.view(batch_size, -1), 1).view(batch_size, 1, 1).expand(batch_size, w, h)

    eps = 2.2204e-16
    s_map = (s_map - mean_s_map) / (std_s_map + eps)

    s_map = torch.sum((s_map * gt).view(batch_size, -1), 1)
    count = torch.sum(gt.view(batch_size, -1), 1)
    return torch.mean(s_map / count)



if __name__ == '__main__':
    main()
