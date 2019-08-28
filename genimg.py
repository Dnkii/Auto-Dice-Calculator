import os
import numpy as np
# import matplotlib
# from matplotlib import pyplot, cm
# import scipy.misc
import cv2
#说明：drawNpGrey（）可以将两个nii.gz文件转换为两个文件夹的png文件供读取训练
import copy
import SimpleITK as sitk
# import skimage.io as io
# import _thread
# import time


t = 0

def read_img(path):
    img = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(img)
    return data

def hu_to_grayscale(volume, hu_min = None, hu_max = None):
    # Clip at max and min values if specified
    if hu_min is not None or hu_max is not None:
        volume = np.clip(volume, hu_min, hu_max)

    # Scale to values between 0 and 1
    mxval = np.max(volume)
    mnval = np.min(volume)
    
    im_volume = (volume - mnval)/max(mxval - mnval, 1e-3)

    # Return values scaled to 0-255 range, but *not cast to uint8*
    # Repeat three times to make compatible with color overlay
    im_volume = 255*im_volume
    #转uint8+增加维度，因为标记如果是彩色则需要RGB三通道，与之相加的CT图也要存成三维数组
    return im_volume
def drawNpGrey(dirname,ArrayDicom,ArrayDicom_mask,start=None,end=None):
    dir="./"+dirname
    if(os.path.exists(dir)==False):
        os.makedirs(dir)
    print(len(ArrayDicom[0, 0]))
    ArrayDicom_mask[np.equal(ArrayDicom_mask,1)] = 255
    ArrayDicom_mask[np.equal(ArrayDicom_mask,2)] = 255
    ArrayDicom = np.transpose(ArrayDicom, (1, 0, 2))
    ArrayDicom_mask = np.transpose(ArrayDicom_mask, (1, 0, 2))
    if start is None:
        start = 0
    if end is None:
        end = len(ArrayDicom[0, 0])
    for i in range(end - start):
        ArrayDicom[:, :,start + i] = hu_to_grayscale(ArrayDicom[:, :,start + i])
        # print(ArrayDicom[:, :,start + i].shape)
        if ArrayDicom[:, :,start + i].shape != (512,512):
            cv2.resize(ArrayDicom[:, :,start + i],(512,512),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite('%s/%05d.png'%(dir,i), cv2.resize(ArrayDicom[:, :,start + i],(512,512),interpolation=cv2.INTER_CUBIC))
            print("Output:"+'%s/%05d.png'%(dir,i))
            cv2.imwrite('%s/%05d_mask.png'%(dir,i), cv2.resize(ArrayDicom_mask[:, :,start + i],(512,512),interpolation=cv2.INTER_CUBIC))
            print("Output:"+'%s/%05d_mask.png'%(dir,i))
        else:
            cv2.imwrite('%s/%05d.png'%(dir,i), ArrayDicom[:, :,start + i])
            print("Output:"+'%s/%05d.png'%(dir,i))
            cv2.imwrite('%s/%05d_mask.png'%(dir,i), ArrayDicom_mask[:, :,start + i])
            print("Output:"+'%s/%05d_mask.png'%(dir,i))
def drawNpGrey_mask(dirname,ArrayDicom,ArrayDicom_mask,start=None,end=None):
    dir="./"+dirname
    if(os.path.exists(dir)==False):
        os.makedirs(dir)
    print(len(ArrayDicom[0, 0]))
    ArrayDicom_mask[np.equal(ArrayDicom_mask,1)] = 178
    ArrayDicom_mask[np.equal(ArrayDicom_mask,2)] = 255
    ArrayDicom = np.transpose(ArrayDicom, (1, 0, 2))
    ArrayDicom_mask = np.transpose(ArrayDicom_mask, (1, 0, 2))
    if start is None:
        start = 0
    if end is None:
        end = len(ArrayDicom[0, 0])
    for i in range(end - start):
        ArrayDicom[:, :,start + i] = hu_to_grayscale(ArrayDicom[:, :,start + i])
        # print(ArrayDicom[:, :,start + i].shape)
        # if np.max(ArrayDicom_mask[:, :,start + i])==0:
        #     continue
        if ArrayDicom[:, :,start + i].shape != (512,512):
            cv2.resize(ArrayDicom[:, :,start + i],(512,512),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite('%s/%05d.png'%(dir,i), cv2.resize(ArrayDicom[:, :,start + i],(512,512),interpolation=cv2.INTER_CUBIC))
            print("Output:"+'%s/%05d.png'%(dir,i))
            cv2.imwrite('%s/%05d_mask.png'%(dir,i), cv2.resize(ArrayDicom_mask[:, :,start + i],(512,512),interpolation=cv2.INTER_CUBIC))
            print("Output:"+'%s/%05d_mask.png'%(dir,i))
        else:
            cv2.imwrite('%s/%05d.png'%(dir,i), ArrayDicom[:, :,start + i])
            print("Output:"+'%s/%05d.png'%(dir,i))
            cv2.imwrite('%s/%05d_mask.png'%(dir,i), ArrayDicom_mask[:, :,start + i])
            print("Output:"+'%s/%05d_mask.png'%(dir,i))

def drawNpGrey_mask_2(dirname,ArrayDicom,ArrayDicom_mask,start=None,end=None):
    dir="./"+dirname
    if(os.path.exists(dir)==False):
        os.makedirs(dir)
    print("总共%s张图片"%len(ArrayDicom[0, 0]))
    # ArrayDicom2 = copy.deepcopy(ArrayDicom)
    for i in range(1,10):
        ArrayDicom_mask[np.equal(ArrayDicom_mask,i)] = 255
        ArrayDicom[np.equal(ArrayDicom,i)] = 255
        # ArrayDicom2[np.equal(ArrayDicom,i)] = 255
    # ArrayDicom2 = ArrayDicom2 - ArrayDicom_mask
    # ArrayDicom[np.equal(ArrayDicom2,5)] = 0
    # ArrayDicom[np.equal(ArrayDicom2,8)] = 0
    # ArrayDicom[np.equal(ArrayDicom,5)] = 255
    # ArrayDicom[np.equal(ArrayDicom,8)] = 255
    ArrayDicom = np.transpose(ArrayDicom, (1, 0, 2))
    ArrayDicom_mask = np.transpose(ArrayDicom_mask, (1, 0, 2))


    if start is None:
        start = 0
    if end is None:
        end = len(ArrayDicom[0, 0])
    for i in range(end - start):
        # ArrayDicom[:, :,start + i] = hu_to_grayscale(ArrayDicom[:, :,start + i])
        # print(ArrayDicom[:, :,start + i].shape)
        # if np.max(ArrayDicom_mask[:, :,start + i])==0:
        #     continue
        # if ArrayDicom[:, :,start + i].shape != (512,512):
        #     cv2.resize(ArrayDicom[:, :,start + i],(512,512),interpolation=cv2.INTER_CUBIC)
        #     cv2.imwrite('%s/%05d.png'%(dir,i), cv2.resize(ArrayDicom[:, :,start + i],(512,512),interpolation=cv2.INTER_CUBIC))
        #     # print("Output:"+'%s/%05d.png'%(dir,i))
        #     cv2.imwrite('%s/%05d_mask0.png'%(dir,i), cv2.resize(ArrayDicom_mask[:, :,start + i],(512,512),interpolation=cv2.INTER_CUBIC))
        #     # print("Output:"+'%s/%05d_mask0.png'%(dir,i))
        #     cv2.imwrite('%s/%05d_mask1.png'%(dir,i), cv2.resize(ArrayDicom_mask2[:, :,start + i],(512,512),interpolation=cv2.INTER_CUBIC))
        #     # print("Output:"+'%s/%05d_mask1.png'%(dir,i))
        # else:
        cv2.imwrite('%s/%05d_mask.png'%(dir,i), ArrayDicom[:, :,start + i])
        # print("Output:"+'%s/%05d_mask.png'%(dir,i))
        cv2.imwrite('%s/%05d_test.png'%(dir,i), ArrayDicom_mask[:, :,start + i])
        # print("Output:"+'%s/%05d_test.png'%(dir,i))
        print("\r",end="",flush = True)
        print("输出第"+"%05d"%i+"张图片中",end="")
        # cv2.imwrite('%s/%05d_mask1.png'%(dir,i), ArrayDicom_mask2[:, :,start + i])
        # print("Output:"+'%s/%05d_mask1.png'%(dir,i))


def nii2png(niiname1, niiname2):
    dirname = "data"#读取路径
    # path_mask1 = '%s/%s.nii.gz'%(path,niiname1) #数据所在路径
    # print(path_mask1)
    data_mask1 = read_img(niiname1)
    # path_mask2 = '%s/%s.nii.gz'%(path,niiname2) #数据所在路径
    data_mask2 = read_img(niiname2)
    drawNpGrey_mask_2("%s"%(dirname),data_mask1,data_mask2)

# for i in range(len(dirlist)-40,len(dirlist)):
#     dirname = dirlist[i]
#     path_mask = './%s/segmentation.nii.gz'%(dirname) #数据所在路径
#     data_mask = read_img(path_mask)
#     path = './%s/imaging.nii.gz'%(dirname) #数据所在路径
#     data = read_img(path)
#     while t>10:
#         pass
#     try:
#         _thread.start_new_thread(drawNpGrey_mask_2,("train5_kidney1tumor2_all/test%s"%(dirname),data,data_mask))
#         print("第%s个线程开始了,正在输出第%s个文件夹"%(t,i))
#     except:
#         print("%s:ERROR"%i)