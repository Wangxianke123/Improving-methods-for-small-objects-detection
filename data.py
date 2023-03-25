import os
import cv2
import numpy as np
import pandas as pd
import math
import random

""" 
一个对小目标数据集使用copy-paste方法进行数据增强的脚本
"""
current_dir = os.path.dirname(__file__)
image_load_dir = os.path.join(current_dir, '..', 'DOTAv1', 'images', 'validation')
image_save_dir = os.path.join(current_dir, 'images', 'val')

label_load_dir = os.path.join(current_dir, '..', 'DOTAv1', 'labels', 'validation')
label_save_dir = os.path.join(current_dir, 'labels', 'val')

class Data:
    def __init__(self, name, label=None, image=None):
        if label is None:
            label = []
        self.name = name
        self.label = label
        self.image = image
        self.shape = image.shape


    def set_name(self, name):
        self.name = name

    def set_img(self, img):
        self.image = img

    def append_label(self,label):
        self.label = label



def load_data(label_dir,image_dir):
    # 获取标注文件的文件名
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    data_list = []
    for f in label_files:
        # 读取标注文件的数据，指定空格为分隔符
        df = pd.read_csv(os.path.join(label_dir, f),dtype = float, header=None, delimiter=' ', names=['class','x', 'y', 'w', 'h'])
        name = f.split(".")[0]
        img =  cv2.imread(os.path.join(image_dir, name +".jpg"))
        
        data = Data(name = name, label = df, image = img)
        data_list.append(data)

    return data_list


def contains_small_obj(data):
    #设置小目标的判定阈值为0.04，面积小于这个值的需要进行后续copy操作
    threshold = 0.04
    temp = data.label

    #筛选出面积小于阈值的目标
    condition = np.sqrt(temp['w'] * temp['h']) < threshold
    temp = temp[condition]

    #若同一类型目标在一张图片中出现大于五次，这里也认为不需要进行增强（相当于复制过了）
    groups = temp.groupby(['class'])
    filtered = groups.filter(lambda x: len(x) <= 3)

    return filtered


def copy_paste(image, label, filterd, name):
    for index, row in filterd.iterrows():
        height, width, channels = image.shape
        #计算需要裁剪的小目标的范围
        x1 = round(width*(row['x']-row['w']/2))
        x2 = round(width*(row['x']+row['w']/2))
        y1 = round(height*(row['y']-row['h']/2))
        y2 = round(height*(row['y']+row['h']/2))
        #w,h是以像素计算的宽，高
        w = x2 - x1
        h = y2 - y1 
        #截取小目标
        crop_img = image[y1:y2, x1:x2]
        #随机生成5个与原图中目标不重合的小目标
        positions = []
        while len(positions) < 5:
            pos = (random.randint(0, width - w), random.randint(0, height - h))
            pos_yolo = yolo_style(pos,w,h,width,height)
            if not overlap(pos_yolo, label):
                positions.append(pos)
                #改写标签
                new_row = {'class': row['class'], 'x': pos_yolo[0], 'y': pos_yolo[1], 'w': pos_yolo[2], 'h': pos_yolo[3]}
                label = label.append(new_row, ignore_index=True)
            #positions = [(random.randint(0, width - w), random.randint(0, height - h)) for _ in range(10)]
        #粘贴图片并重写标注
        for pos in positions:
            image[pos[1]:pos[1]+h, pos[0]:pos[0]+w] = crop_img
        label['class'] = label['class'].astype(int).apply(lambda x: int(x))
        label.to_csv(os.path.join(label_save_dir,name+'_aug.txt'), sep=' ', index=False, header=False)
        cv2.imwrite(os.path.join(image_save_dir, name+'_aug.jpg'), image)

def yolo_style(pos,w,h,width,height):
    x = (pos[0] + w/2)/ width
    y = (pos[1] + h/2)/ height
    w1 = w/width
    h1 = h/height
    pos_yolo = [x,y,w1,h1]
    return pos_yolo

def overlap(pos_yolo,label):
    for index, row in label.iterrows():
        x_min = max(pos_yolo[0]-pos_yolo[2]/2, row['x']-row['w']/2)
        y_min = max(pos_yolo[1]-pos_yolo[3]/2, row['y']-row['h']/2)
        x_max = min(pos_yolo[0]+pos_yolo[2]/2, row['x']+row['w']/2)
        y_max = min(pos_yolo[1]+pos_yolo[3]/2, row['y']+row['h']/2)

        x = max(0, x_max - x_min)
        y = max(0, y_max - y_min)
        if x * y > 0:
            return True
    return False

def main():
    data_list = load_data(label_load_dir, image_load_dir)
    for data in data_list:
        filterd = contains_small_obj(data)
        if filterd.empty:
            print("can't find small object in file:",data.name)
            data.label.to_csv(os.path.join(label_save_dir,data.name+'.txt'), sep=' ', index=False, header=False)
            cv2.imwrite(os.path.join(image_save_dir, data.name+'.jpg'), data.image)
        else:
            print("find small object in file:",data.name)
            data.label.to_csv(os.path.join(label_save_dir,data.name+'.txt'), sep=' ', index=False, header=False)
            cv2.imwrite(os.path.join(image_save_dir, data.name+'.jpg'), data.image)
            copy_paste(data.image,data.label,filterd,data.name)

if __name__ == "__main__":
    main()
