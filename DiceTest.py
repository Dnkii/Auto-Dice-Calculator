# import numpy as np
import torch
import PIL.Image as Image
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
# import scipy.misc
import os
from loss import DiceLoss
from genimg import nii2png
from genimgforwang import nii2png_wang

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_transforms = transforms.Compose([
    transforms.ToTensor()
])

# mask只需要转换为tensor
y_transforms = transforms.Compose([
    transforms.ToTensor(),
])

#参数解析
#收集原图和带标记的图
def make_dataset1(root):
    test = []
    img = []
    mask = []
    a=0
    for dirName,subdirList,fileList in os.walk(root):
        for filename in fileList:
            if "mask.png" in filename.lower(): #判断文件是否为dicom文件
                mask.append(os.path.join(dirName,filename)) # 加入到列表中
            else:
                if "test.png" in filename.lower():
                    test.append(os.path.join(dirName,filename))
                else:
                    if ("mask" not in filename.lower()) and ("test" not in filename.lower()):
                        img.append(os.path.join(dirName,filename))
    # print("共%s张dcm图片"%len(img))
    return test,mask
#只收集原图

#返回原图和mask图
class LiverDataset1(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs, masks = make_dataset1(root)
        self.imgs = imgs
        self.masks = masks
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_x = Image.open(self.imgs[index])
        img_y = Image.open(self.masks[index])
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)



def test_model(criterion, dataload, num_epochs=1):
    Loss_list = []
    Accuracy_list = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        epoch_loss1 = 0
        epoch_loss2 = 0
        step = 0
        for x, y in dataload:
            step += 1
            outputs = x.to(device)
            labels = y.to(device)
            loss1,loss2 = criterion(outputs, labels)
            epoch_loss1 += loss1.item()
            epoch_loss2 += loss2.item()
            epoch_loss = loss1/loss2
            print("%d/%d,test_loss:%0.6f" % (step, (dt_size - 1) // dataload.batch_size + 1, epoch_loss))
        Loss_list.append(epoch_loss / ((dt_size - 1)// dataload.batch_size + 1))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss1/epoch_loss2))

    return


def test_plot(path="data", num_epochs=1, start=0, end=0):
    criterion = DiceLoss()
    liver_dataset = LiverDataset1(path, transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_model(criterion, dataloaders, num_epochs)


if __name__ == '__main__':
    while 1:
        # dirname = input("请输入文件所在文件夹地址,若为当前文件夹下请输入‘.’")
        x = input("选择工作模式，标准测试输入‘1’，排除多余测试输入‘2’:")
        file1 = input("请输入真值文件(请使用英文路径):")
        file2 = input("请输入分割文件(请使用英文路径):")
        
        if x =='1':
            nii2png(file1, file2)
        else:
            nii2png_wang(file1, file2)
        test_plot("data")