VOCroot = '/home/ipsg/code/sx/datasets'
# VOCroot = 'E:/datasets'
# DATASET = 'voc_smallest_withneg'
# DATASET = 'voc_smallest'
# DATASET = 'voc_2007'
# DATASET = 'voc_tiny_withneg'
DATASET = 'nuclear_cold_tiny_withneg'

COLORS_BGR = [(0,0,0), (0,0,0), (100,101,3),(154, 157, 252), (3,155,230), (78,73,209), (148,137,69)]
NUCLEAR_COLD = [
    'robot',
    'battery',
    'wood',
    'brick',
    'barrel',
    'box',
    'sack',
    'motor'
]
NUCLEAR_COLD_TINY = [
    'robot',
    'battery',
    'brick',
    'barrel',
    'box',
]
NUCLEAR_REAL = [
    'cylinder',
    'robot',
    'wagon',
    'hook',
    'sack',
]
VOC_SMALLEST = [
    'aeroplane',
    'bicycle',
    'boat',
    'bus',
    'car',
]
VOC_TINY = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
]
VOC_2007 = ('aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')

VOC_CLASSES = []
if 'voc_smallest' in DATASET:
    VOC_CLASSES = VOC_SMALLEST
if 'voc_tiny' in DATASET:
    VOC_CLASSES = VOC_TINY
if 'nuclear_cold' in DATASET:
    VOC_CLASSES = NUCLEAR_COLD
if 'nuclear_real' in DATASET:
    VOC_CLASSES = NUCLEAR_REAL
if 'voc_2007' == DATASET:
    VOC_CLASSES = VOC_2007
if 'nuclear_cold_tiny' in DATASET:
    VOC_CLASSES = NUCLEAR_COLD_TINY

# Example: VOC_CLASSES = ('__background__', 'neg', 'car', 'person','zebra_crossing')
VOC_CLASSES.insert(0, '__background__')
if 'withneg' in DATASET:
    VOC_CLASSES.insert(1, 'neg')

rgb_means = (104, 117, 123)
p = 0.6
num_classes = len(VOC_CLASSES)
weight_decay = 0.0005
gamma = 0.1
momentum = 0.9
top_k = 200

FACE_240 = {
    'feature_maps': [30, 15, 8, 4],
    'min_dim': 240,
    'steps': [8, 16, 30, 60],
    'min_sizes': [20, 30, 40, 50],
    'max_sizes': [30, 40, 50, 60],
    'aspect_ratios': [[2], [2], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'anchor_level_k': [3, 3, 3, 3]
}
VEHICLE_240 = {
    'feature_maps': [30, 15, 8, 4],
    'min_dim': 240,
    'steps': [8, 16, 30, 60],
    'min_sizes': [30, 80, 130, 180],
    'max_sizes': [80, 130, 180, 240],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    'variance': [0.1, 0.2],
    'clip': True,
    'anchor_level_k': [3, 3, 3, 3]
}
DetNet_300 = {
    'feature_maps': [38, 19, 10, 10, 10, 10],
    'min_dim': 300,
    'steps': [8, 16, 32, 32, 32, 32],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    'variance': [0.1, 0.2],
    'clip': True,
}
MobileNetV1_300 = {
    'feature_maps': [19, 10, 5, 3, 2, 1],
    'min_dim': 300,
    'steps': [16, 30, 60, 100, 150, 300],
    'min_sizes': [60, 105, 150, 195, 240, 285],
    'max_sizes': [105, 150, 195, 240, 285, 315],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    'variance': [0.1, 0.2],
    'clip': True,
}
RefineDet_320 = {
    'feature_maps': [40, 20, 10, 5],
    'min_dim': 320,
    'steps': [16, 30, 60, 100, 150, 300],
    'min_sizes': [60, 120, 180, 240],
    'max_sizes': [120, 180, 240, 300],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3]],
    'variance': [0.1, 0.2],
    'clip': True,
}
RefineDet_320_V2 = {
    'feature_maps': [40, 20, 10, 5, 3, 1],
    'min_dim': 320,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    'variance': [0.1, 0.2],
    'clip': True,
}
VOC_300 = {
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [20, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    'variance': [0.1, 0.2],
    'clip': True,
    # 'anchor_level_k': [3,3,3,3,3,3]
    'anchor_level_k': [9, 9, 9, 9, 9, 9]
}
official_VOC_300 = {
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
}
VOC_512 = {
    'feature_maps': [64, 32, 16, 8, 4, 2],
    'min_dim': 512,
    'steps': [8, 16, 32, 64, 128, 256],
    'min_sizes': [35.84, 76.8, 153.6, 230.4, 307.2, 384.0],
    'max_sizes': [76.8, 153.6, 230.4, 307.2, 384.0, 512.0],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    'variance': [0.1, 0.2],
    'clip': True,
}
COCO_300 = {
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
}
COCO_512 = {
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],
    'min_dim': 512,
    'steps': [8, 16, 32, 64, 128, 256, 512],
    'min_sizes': [20.48, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8],
    'max_sizes': [51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
}
COCO_mobile_300 = {
    'feature_maps': [19, 10, 5, 3, 2, 1],
    'min_dim': 300,
    'steps': [16, 32, 64, 100, 150, 300],
    'min_sizes': [45, 90, 135, 180, 225, 270],
    'max_sizes': [90, 135, 180, 225, 270, 315],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
}
M2Det_320 = {
    'feature_maps': [40, 20, 10, 5, 3, 1],
    'min_dim': 320,
    'steps': [8, 16, 32, 64, 110, 320],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    'variance': [0.1, 0.2],
    'clip': True,
}
