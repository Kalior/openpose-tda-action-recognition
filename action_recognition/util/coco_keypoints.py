from enum import Enum


class COCOKeypoints(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18

coco_connections = [(k1.value, k2.value) for k1, k2 in [
    (COCOKeypoints.Nose,        COCOKeypoints.Neck),
    (COCOKeypoints.Neck,        COCOKeypoints.RShoulder),
    (COCOKeypoints.Neck,        COCOKeypoints.LShoulder),
    (COCOKeypoints.Neck,        COCOKeypoints.RHip),
    (COCOKeypoints.Neck,        COCOKeypoints.LHip),
    (COCOKeypoints.RShoulder,   COCOKeypoints.RElbow),
    (COCOKeypoints.RElbow,      COCOKeypoints.RWrist),
    (COCOKeypoints.LShoulder,   COCOKeypoints.LElbow),
    (COCOKeypoints.LElbow,      COCOKeypoints.LWrist),
    (COCOKeypoints.RHip,        COCOKeypoints.RKnee),
    (COCOKeypoints.RKnee,       COCOKeypoints.RAnkle),
    (COCOKeypoints.LHip,        COCOKeypoints.LKnee),
    (COCOKeypoints.LKnee,       COCOKeypoints.LAnkle)
]]
