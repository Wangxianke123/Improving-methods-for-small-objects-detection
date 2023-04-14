
from turtle import shape
import matplotlib.pyplot as plt
import dis
import os
import random
import numpy as np
from scipy.cluster.vq import kmeans,whiten

np.seterr(divide='ignore',invalid='ignore')

label_dir = '/home/Wuyong/wenkai/my_yolo/datasets/DOTAv1/labels/train'
#label_dir = '/home/Wuyong/wenkai/my_yolo/datasets/coco128/labels/train2017'
label_dir = '/home/Wuyong/wenkai/my_yolo/datasets/TGRS/TGRS-HRRSD-Dataset/OPT2017/yolo_style'
image_size = 1024
def load_data(label_dir):
    """ 取出对应目录下所有Yolo格式的标签
        返回形如[[w1,h1],[w2,h2].....[wn,hn]]的数组 """
    wh_values = [] # 存储w和h值的数组
    # 遍历目录下所有的.txt文件
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(label_dir, filename)
            # 读取文件中每一行的内容
            with open(file_path, "r") as f:
                for line in f:
                    # 解析每行内容中的w和h值
                    values = line.strip().split(" ")
                    wh_values.append((float(values[3]), float(values[4])))
    # 输出所有w和h值
    return np.array(wh_values)

def iou(box, clusters):
    """ 计算box和cluster的iou
        输出规模为(box.shape[0],cluster.shape[0]) """
    #计算相交区域的面积
    x = np.minimum(box[:,0],clusters[:,0].reshape(clusters.shape[0],1)).T
    y = np.minimum(clusters[:,1].reshape(clusters.shape[0],1),box[:,1]).T
    intersection = x*y
    #分布计算box和cluster中每个单元的面积
    area1 = (box[:,0] * box[:,1]).reshape(box.shape[0],1)
    area2 = (clusters[:,0]*clusters[:,1]).reshape(1,clusters.shape[0])
    #利用广播操作得到IoU矩阵
    iou = intersection / (area1 + area2 -intersection + 1e-8 )
    return iou

def new_clus(box,min_indices,k):
    """ 根据已经分类好的Box得到新的聚类中心
        返回值为新的clusters数组 """
    
    means = [0] * k
    for index in range(k):
        # 使用布尔索引将分类为index的数据取出来
        label_data = box[min_indices == index]
        # 计算均值并添加到列表中
        means[index] = np.mean(label_data, axis=0)
    means = np.array(means)
    return means
    
def kmeans(box, k):
    """ 采取经典的k-means算法获得k个box的聚类 """
    #随机指定k个box作为聚类中心
    clusters = box[np.random.choice(box.shape[0], k, replace=False)]
    distance = np.zeros((box.shape[0],k))
    while True: 
        #计算每个box与聚类中心的距离,定义distance(box,centroid)=1-iou(box,centroid)
        # 详见:'Yolo9000:Better,Faster,Stronger'
        distance = 1-iou(box,clusters)
        #将每个Box分类到与其距离最小的聚类中
        min_indices = np.argmin(distance, axis=1)
        #对每个聚类计算新的聚类中心
        temp = new_clus(box,min_indices,k)
        if(clusters==temp).all():
            break
        clusters = temp
    return clusters

def average_iou(box,clusters):
    """ 返回聚类之后的平均iou """
    return np.mean(np.max(iou(box,clusters),axis=1))

def kmeans_pp(box, k):
    """ 改进版的k-means算法,kmeans++ """
    #随机选择一个点作为聚类中心
    clusters=box[np.random.choice(box.shape[0], 1, replace=False)]
    clusters = np.array(clusters)
    for _ in range(1,k):
        #对集合中的每个点，计算其与各个聚类中心最小的距离distance(box,centroid)=1-iou(box,centroid)
        distance = np.min((1-iou(box,clusters)), axis = 1)
        #以轮盘方式，按照概率p = D(x)^2/sum(D(x)^2)的方式获得下一个聚类中心
        D = np.sum(np.power(distance, 2))
        prob = np.power(distance, 2) / D
        index = np.random.choice(np.arange(box.shape[0]), 1, p=prob) 
        #将新产生的聚类中心加入
        new_cluster = box[index]
        clusters=np.vstack([clusters,new_cluster])
    return  clusters

def kmeans_mc2(box, k, chain_length = 1000):
    """ 改进版的k-means++算法，基于马尔科夫链蒙特卡洛采样方法进行新的聚类中心的生成。
        Paper原文:Fast and Provably Good Seedings for k-Means,
        https://proceedings.neurips.cc/paper/2016/file/d67d8ab4f4c10bf22aa353e27879133c-Paper.pdf
    """
    n = box.shape[0]#number of boxes
    #随机选择一个点作为聚类中心
    clusters=box[np.random.choice(n, 1, replace=False)]
    clusters = np.array(clusters)
    #获得采样所依据的概率分布这里用的是P(x) =  1/2 * D(x)^2/sum(D(x)^2) + 1/2n
    dist = np.power(np.min((1-iou(box,clusters)), axis = 1), 2)
    prob = np.power(dist, 2) / np.sum(np.power(dist, 2)) * 0.5 + 0.5 / n
    for _ in range(1,k):
        #生成新的样本中心
        index = np.random.choice(np.arange(box.shape[0]), 1, p=prob) 
        x = box[index]
        dx = dist[index]
        #MC过程
        for __ in range(1,chain_length): 
            index_y = np.random.choice(np.arange(box.shape[0]), 1, p=prob) 
            y = box[index_y]
            dy = dist[index_y]
            if dy*prob[index]/dx*prob[index_y] > random.random():
                x = y
                dx = dy
        clusters=np.vstack([clusters,x])
    return clusters

clasic_cluster = np.array([
    [16,16],
    [21,36],
    [38,20],
    [32,41],
    [44,16],
    [39,77],
    [69,49],
    [78,83],
    [138,114],
    [117,195],
    [301,269],
    [620,592]
])
def main():
    k=12
    box = load_data(label_dir)
    iou_kmeans=[]
    iou_kmeans_pp=[]
    iou_kmeans_mc2 = []
    for k in range(3,16):
        clusters = kmeans(box,k)
        average = average_iou(box,clusters)
        iou_kmeans.append(average)
        print('average iou of kmeans: in k=',k,':',average)

        clusters = kmeans_pp(box,k)
        average = average_iou(box,clusters)
        iou_kmeans_pp.append(average)
        print('average iou of kmeans++: in k=',k,':',average)
    
        clusters = kmeans_mc2(box, k, 500)
        average = average_iou(box,clusters)
        iou_kmeans_mc2.append(average)
        print('average iou of kmeans-mc2: in k=',k,':',average)
    
        print('-------------------')
        print(' ')
    
    print('iou_kmeans:',iou_kmeans)
    print('iou_kmeans_pp:',iou_kmeans_pp)
    print('iou_kmeans_mc2:',iou_kmeans_mc2)

    k_values = list(range(3, 16))
    plt.plot(k_values, iou_kmeans, label='kmeans')
    plt.plot(k_values, iou_kmeans_pp, label='kmeans++')
    plt.plot(k_values, iou_kmeans_mc2, label='kmeans_mc2')
    # 设置图像标题和坐标轴标签
    plt.title('Performance Comparison')
    plt.xlabel('k')
    plt.ylabel('Performance')

    # 添加图例
    plt.legend()

    plt.savefig('performance.png')

"""  iou_kmeans = [0.5415150494644347, 0.557664416263247, 0.593047337041616, 0.6242709090819557, 0.6234807773246887, 0.6434684382700065, 0.6605868991084154, 0.6707029225347229, 0.6905788630210125, 0.6993638004037892, 0.7061849493270493, 0.7177598556489148, 0.7196761997955391]
    iou_kmeans_pp = [0.4756547024041466, 0.5436220261475662, 0.57018798622981, 0.6101667929241068, 0.5992031985770481, 0.6383265552498404, 0.6417872296256225, 0.6488833627524041, 0.6719021577928902, 0.6779125956241345, 0.6861291225775827, 0.6946317339214223, 0.6979452854517735]
    k_values = list(range(3, 16))
    plt.plot(k_values, iou_kmeans, label='Algorithm kmeans')
    plt.plot(k_values, iou_kmeans_pp, label='Algorithm kmeans++')

    # 设置图像标题和坐标轴标签
    plt.title('Performance Comparison')
    plt.xlabel('k')
    plt.ylabel('Performance')

    # 添加图例
    plt.legend()

    plt.savefig('performance.png') """
    

if __name__ == "__main__":
    main()
