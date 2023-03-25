from concurrent.futures import thread
import torch
import torchvision
import copy
from math import exp
# 定义一组检测框，包括位置信息和得分
new_tensor = torch.tensor([
    [100, 100, 200, 200, 0.3],
    [150, 150, 250, 250, 0.9],
    [100, 150, 200, 250, 0.4],
    [300, 300, 400, 400, 0.7],
])

boxes = new_tensor[:,:4]
scores = new_tensor[:,4]
# print(boxes)
# print(scores)

# print(indices)

def my_nms(boxes, scores, threshold = 0.3):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    #分别计算每个box的面积，为计算IoU作准备
    areas = (x2-x1)*(y2-y1) 
    _,indices = scores.sort(descending=True)
    keep = []
    while len(indices)>0:
        #取置信度最大的index，将其放入keep
        i = indices[0]
        keep.append(i)
        if len(indices)==1:
            break
        else:
            indices = indices[1:]
        #clamp方法可以可以使tensor中小于或大于指定值的数据变为指定的边界,
        #使用数组操作代替for循环，增加并行度
        xx1 = x1[indices].clamp(min=x1[i])   # xx1=max(x1,x1')
        yy1 = y1[indices].clamp(min=y1[i])
        xx2 = x2[indices].clamp(max=x2[i])   #xx2=min(x2,x2')
        yy2 = y2[indices].clamp(max=y2[i])
        overlap = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)

        iou = overlap / (areas[i]+areas[indices]-overlap) 
        kept = torch.where(iou < threshold)[0]
        if len(kept)==0:
            break
        indices = indices[kept]
    return torch.LongTensor(keep)


def soft_nms(boxes, scores, iou_thres = 0.3, gaussian = False, score_thres = 0.25, sigma=0.5):
    """ 
    soft_nms的实现,详见paper：
    'Soft-NMS-Improving Object Detection With One Line Of Code'
    经验证，该方法普遍可以在不改变网络结构和计算复杂度的条件下实现约2%的AP提升
    boxes:输入的bounding box,格式为x1y1x2y2
    score:对应box的置信度
    iou_thres:nms时用到的IoU阈值
    gaussian:采用高斯方法还是线性方法进行抑制
    score_thres:soft_nms时用到的分数阈值，低于这个阈值的score会被认为是0
    sigma:高斯抑制时的参数
    
    Return：NMS后有效检测框的索引
    """
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    new_scores = copy.deepcopy(scores)
    #分别计算每个box的面积，为计算IoU作准备
    areas = (x2-x1)*(y2-y1) 
    keep = []
    while len(new_scores)>0:
        i = torch.argmax(new_scores)
        if new_scores[i]<score_thres:
            break
        keep.append(i)
        #clamp方法可以可以使tensor中小于或大于指定值的数据变为指定的边界,
        #使用数组操作代替for循环，增加并行度
        xx1 = x1.clamp(min=x1[i])   # xx1=max(x1,x1')
        yy1 = y1.clamp(min=y1[i])
        xx2 = x2.clamp(max=x2[i])   #xx2=min(x2,x2')
        yy2 = y2.clamp(max=y2[i])
        overlap = (xx2-xx1).clamp(min=0) * (yy2-yy1).clamp(min=0)
        iou = overlap / (areas[i]+areas-overlap) 
        #soft nms
        changed = torch.where(iou >= iou_thres)[0]
        if gaussian:
            new_scores[changed] = new_scores[changed] * torch.exp(-torch.pow(iou[changed],2)/sigma)
        else:
            new_scores[changed] = new_scores[changed]*(1-iou[changed])
    return torch.LongTensor(keep).sort()[0]


keep1 = my_nms(boxes, scores, threshold=0.1)
print(keep1)
keep2 = torchvision.ops.nms(boxes, scores, iou_threshold=0.1)
print(keep2)


keep3 = soft_nms(boxes, scores, iou_thres = 0.1, score_thres=0.25, gaussian=True)
print(keep3)

