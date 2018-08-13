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

coco_connections = [
    (COCOKeypoints.Nose.value, COCOKeypoints.Neck.value),
    (COCOKeypoints.Neck.value, COCOKeypoints.RShoulder.value),
    (COCOKeypoints.Neck.value, COCOKeypoints.LShoulder.value),
    (COCOKeypoints.Neck.value, COCOKeypoints.RHip.value),
    (COCOKeypoints.Neck.value, COCOKeypoints.LHip.value),
    (COCOKeypoints.RShoulder.value, COCOKeypoints.RElbow.value),
    (COCOKeypoints.RElbow.value, COCOKeypoints.RWrist.value),
    (COCOKeypoints.LShoulder.value, COCOKeypoints.LElbow.value),
    (COCOKeypoints.LElbow.value, COCOKeypoints.LWrist.value),
    (COCOKeypoints.RHip.value, COCOKeypoints.RKnee.value),
    (COCOKeypoints.RKnee.value, COCOKeypoints.RAnkle.value),
    (COCOKeypoints.LHip.value, COCOKeypoints.LKnee.value),
    (COCOKeypoints.LKnee.value, COCOKeypoints.LAnkle.value)
]
