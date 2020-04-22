# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 21:06:36 2017

@author: Rotem Goren
"""

import os #do directory operations
import pandas as pd #noce for data analysis
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
#import tensorflow as tf
import glob
import gzip
import nibabel as nib
from Models import VNet,Dataset
import torch
from torch import nn
from collections import OrderedDict
from tqdm import tqdm
import time

#from mask_functions import mask2rle,rle2mask


IMG_PX_SIZE =512


data_dir='D:\\Task06_Lung\\Task06_Lung'
#labels_df=pd.read_csv('D:/siim-acr-pneumothorax-segmentation/train-rle-sample.csv',names=['Id','Values'],index_col=False)
#labels_df.head()
def focal_loss(alpha=0.25, gamma=2.0):
    def focal_crossentropy(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)

        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))

        alpha_factor = 1
        modulating_factor = 1

        alpha_factor = y_true * alpha + ((1 - alpha) * (1 - y_true))
        modulating_factor = K.pow((1 - p_t), gamma)

        # compute the final loss and return
        return K.mean(alpha_factor * modulating_factor * bce, axis=-1)

    return focal_crossentropy

def chunks(l,n):
    for i in range(0,len(l),n):
        yield l[i:i+n]




def preprocess_data(patient,IMG_PX_SIZE=50,visulize=False):
    #label=labels_df[labels_df['Id']==patient[:-4]]['Values'].values[0]
    try:
        img = nib.load(imagesTr_path+patient)
        img_data=img.get_fdata()
        #img_data = img_data[:, :, int(img_data.shape[2]/2) - int(SLICES / 2):int(img_data.shape[2]/2) + int(SLICES / 2)]


        img_data = cv2.resize(img_data, (IMG_PX_SIZE, IMG_PX_SIZE))

        img_data = (img_data-np.min(img_data,axis=(0,1)))/(np.max(img_data,axis=(0,1))-np.min(img_data,axis=(0,1)))
        for i in range(img_data.shape[2]):
            img_data[:, :, i] = cv2.equalizeHist((img_data[:, :, i]*255).astype(np.uint8))

        img_data = (img_data - np.min(img_data, axis=(0, 1))) / (np.max(img_data, axis=(0, 1)) - np.min(img_data, axis=(0, 1)))
        img_data = np.array(img_data,dtype=np.float32)
        img = nib.load(labelsTr_path+patient)
        label_data=img.get_fdata()
        #label_data = label_data[:, :, int(label_data.shape[2]/2) - int(SLICES / 2):int(label_data.shape[2]/2) + int(SLICES / 2)]

        label_data = cv2.resize(label_data, (IMG_PX_SIZE, IMG_PX_SIZE))

        #label=[np.zeros_like(label_data),np.zeros_like(label_data)]
        #label[0][label_data==0] = 1
        #label[1][label_data!=0] = 1
        label=label_data
        label=np.array(label,dtype=np.float32)
        if(visulize==True):
            #while(True):
            #y = fig.add_subplot(4, 4, i+1)
            for i in range(img_data.shape[2]):

                cv2.imshow('im',np.array(img_data[ :, :, i]*255,dtype=np.uint8))
                cv2.waitKey(10)

                cv2.imshow('label',np.array(label_data[ :, :, i]*255,dtype=np.uint8))
                cv2.waitKey(10)


        return img_data,label
    except:
        return [],[]
    #new_slices = []
    #slices=[cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE)) for each_slice in slices]

def DiceLoss(x,y):
    numerator = 2 * torch.sum(x * y, dim=(2,3,4))
    denominator = torch.sum(x + y, dim=(2,3,4))
    #eps = torch.FloatTensor([1e-6]).to(device)
    return 1 - numerator / denominator

def train_model(x_train,y_train,x_valid,y_valid):
    torch.manual_seed(0)
    ngpu=1
    BATCH_SIZE=2
    EPOCHS=500
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    model_file='VNet.pth'

    model = VNet(device)
    if (os.path.isfile(model_file)):
        state_dict = torch.load(model_file, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if (k[:7] == 'module.'):
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v
        # load params
        model.load_state_dict(new_state_dict)

    if (device.type == 'cuda') and (ngpu > 1):
        model = nn.DataParallel(model, list(range(ngpu)), dim=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))

    for epoch in range(EPOCHS):
        train_loss = 0
        valid_loss = 0
        train_accuracy=0
        valid_accuracy=0
        model.train()
        start = time.time()
        i=0
        for x,y in tqdm(zip(x_train,y_train)):

            #try:
                dataset = Dataset(x, y)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE*ngpu,
                                                               shuffle=False, num_workers=1,
                                                               pin_memory=True)  # divided into batches
                optimizer.zero_grad()
                for (X,gt) in (dataloader):
                    if((gt.sum(dim=(2,3,4))>0).all()):
                        #start = time.time()
                        i+=1
                        X=X.to(device, non_blocking=True).float()
                        gt=gt.to(device, non_blocking=True).float()

                        o = model(X)
                        weight = (gt.shape[1:].numel() - gt.sum(dim=(1, 2, 3, 4))) / gt.sum(dim=(1, 2, 3, 4))
                        weight = weight.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
                        loss = nn.BCELoss(weight=weight)(o,gt)

                        #loss = DiceLoss(o,gt)


                        optimizer.zero_grad()
                        loss.backward()


                        train_loss += loss.item()
                        #if (i % int(len(dataloader)/10) == 0):
                        optimizer.step()



                        #predicted = torch.argmax(o.data, 1).sum()
                        #true = torch.argmax(gt.data, 1).sum()

                        predicted = o.data
                        predicted[predicted < 0.5] = 0
                        predicted[predicted > 0.5] = 1
                        true = gt.data

                        train_accuracy+= (predicted*true).sum()/ true.sum()

                        torch.cuda.empty_cache()

            #except:
            #    pass
        train_accuracy = train_accuracy / i
        train_loss = train_loss / i
        torch.cuda.empty_cache()


        #model.eval()
        i=0
        with torch.no_grad():
            for x, y in tqdm(zip(x_valid, y_valid)):
                #try:
                    dataset = Dataset(x, y)
                    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE*ngpu,
                                                             shuffle=False, num_workers=1,
                                                             pin_memory=True)  # divided into batches

                    for X, gt in dataloader:
                        if ((gt.sum(dim=(2,3,4))>0).all()):
                            i+=1
                            X = X.to(device).float()
                            gt = gt.to(device).float()

                            o = model(X)
                            #loss = nn.BCELoss()(o,gt)
                            weight = (gt.shape[1:].numel() - gt.sum(dim=(1, 2, 3, 4))) / gt.sum(dim=(1, 2, 3, 4))
                            weight = weight.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
                            loss = nn.BCELoss(weight=weight)(o, gt)
                            #loss = DiceLoss(o,gt)


                            valid_loss += loss.item()

                            #predicted = torch.argmax(o.data, 1).sum()
                            #true = torch.argmax(gt.data, 1).sum()

                            predicted = o.data
                            predicted[predicted < 0.5] = 0
                            predicted[predicted > 0.5] = 1
                            true = gt.data
                            valid_accuracy += (predicted * true).sum() / true.sum()

                            torch.cuda.empty_cache()
                #except:
                #    pass
            valid_accuracy = valid_accuracy/i
            valid_loss = valid_loss / i
            torch.cuda.empty_cache()
        print("Est time={} sec".format(time.time() - start))

        print('[{}/{}]\tTrain loss: {}\t Train Acc: {}\tValid loss:{}\t Valid Acc:{}'.format(epoch, EPOCHS, train_loss,
                                                                                             train_accuracy, valid_loss,
                                                                                             valid_accuracy))
        torch.save(model.state_dict(), '%s' % model_file)

def test_model(x_test,y_test,visulize = True):
    torch.manual_seed(0)
    ngpu=1
    BATCH_SIZE=1
    EPOCHS=50
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    model_file='VNet.pth'

    model = VNet(device)
    if (os.path.isfile(model_file)):
        state_dict = torch.load(model_file, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if (k[:7] == 'module.'):
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v
        # load params
        model.load_state_dict(new_state_dict)

    if (device.type == 'cuda') and (ngpu > 1):
        model = nn.DataParallel(model, list(range(ngpu)), dim=0)

    model.train()
    with torch.no_grad():
        for x, y in tqdm(zip(x_test, y_test)):

            # try:
            dataset = Dataset(x, y,3)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=ngpu,
                                                     shuffle=False, num_workers=1,
                                                     pin_memory=True)  # divided into batches

            for (X, gt) in (dataloader):
                #if (gt[:, 0, :, :, :].sum() > 0):
                    # start = time.time()

                    X = X.to(device, non_blocking=True).float()
                    gt = gt.to(device, non_blocking=True).float()

                    o = model(X)
                    #loss = nn.BCELoss()(o, gt)

                    #loss.backward()

                    predicted = o.data
                    predicted[predicted < 0.5] = 0
                    predicted[predicted > 0.5] = 1
                    true = gt.data


                    visulize = True
                    if (visulize == True):
                        # while(True):
                        # y = fig.add_subplot(4, 4, i+1)
                        alpha=0.5
                        x = X.cpu().numpy()
                        y = true.cpu().numpy()
                        predict = predicted.cpu().numpy()

                        for j in range(x.shape[4]):

                            gt_mask=np.zeros((IMG_PX_SIZE,IMG_PX_SIZE,3),dtype=np.uint8)
                            gt_mask[:,:,2]=np.array(y[0, 0, :, :, j] * 255, dtype=np.uint8)
                            gt_image = alpha*np.repeat(np.expand_dims(x[0,0,:,:,j]*255,axis=2),3,axis=2) + (1-alpha)*gt_mask
                            cv2.imshow('gt', np.array(gt_image,dtype=np.uint8))
                            cv2.waitKey(10)

                            predict_mask=np.zeros((IMG_PX_SIZE,IMG_PX_SIZE,3),dtype=np.uint8)
                            predict_mask[:,:,2]=np.array(predict[0, 0, :, :, j] * 255, dtype=np.uint8)
                            predict_image = alpha*np.repeat(np.expand_dims(x[0,0,:,:,j]*255,axis=2),3,axis=2) + (1-alpha)*predict_mask
                            cv2.imshow('predict', np.array(predict_image,dtype=np.uint8))
                            cv2.waitKey(10)
                            if(gt_mask.sum()>0):
                                print("tumor")



                    # total_train += true.size(0) * true.size(1) * true.size(2)*true.size(3)
                    # correct_train += predicted.eq(true.data).sum().item()
                    torch.cuda.empty_cache()


if __name__ == "__main__":

    data = []
    label = []

    imagesTr_path=data_dir+'\\imagesTr\\'
    labelsTr_path = data_dir + '\\labelsTr\\'
    if (os.path.isfile(data_dir+'\\train_data.npy')):
        train_data=np.load(data_dir + '\\train_data.npy')
        train_label=np.load(data_dir + '\\train_label.npy')

    else:
        for num, patientFile in enumerate(glob.glob(os.path.join(imagesTr_path, "*.gz"))[20:30]):

            patient = patientFile.split('\\')[-1]
            if (num % 10 == 0):
                print(num)
            try:

                img_data, label_data = preprocess_data(patient, IMG_PX_SIZE, visulize=False)
                if img_data != []:
                    data.append(img_data)
                    label.append(label_data)

            except KeyError as e:
                print('This is unlabeld data')


        #np.save(data_dir + '\\train_data.npy',train_data)
        #np.save(data_dir + '\\train_label.npy', train_label)


    vaildRatio=0.8
    train_data = data[:int(len(data)*vaildRatio)]
    train_label = label[:int(len(label) * vaildRatio)]

    vaild_data = data[int(len(data)*vaildRatio):]
    vaild_label = label[int(len(label) * vaildRatio):]

    train_model(train_data,train_label,vaild_data,vaild_label)
    test_model(train_data,train_label,visulize = True)

    # img_data=torch.FloatTensor(img_data).unsqueeze(0).unsqueeze(1)
    # model=VNet()
    # y=model(img_data)

    #np.save('muchdata-{}-{}-{}.npy'.format(IMG_PX_SIZE,IMG_PX_SIZE,HM_SLICES),much_data)


