# Improving-methods-for-small-object-detection
本科毕业设计相关，一些针对小目标检测的优化方法

1.nms_test.py:
    soft-nms 的实现，详见paper：'Soft-NMS-Improving Object Detection With One Line Of Code'
    

2.data.py:
    一个对小目标数据集进行数据增强的脚本，使用copy-paste策略。
    详见paper:'Augmentation for small object detection'

3.my_kmeans:
    针对yolov3中anchor选取所采用的kmeans算法的优化，实际测试效果会根据数据集不同而有所变化
