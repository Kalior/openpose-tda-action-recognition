import numpy as np

from .coco_keypoints import COCOKeypoints
# ^ Should be the same as tf_pose.common.CocoPart
# from tf_pose import common


def tf_openpose_human_to_np(human, image_width, image_height):
    # The same transformation that the authors are doing for their plotting:
    # https://github.com/ildoonet/tf-pose-estimation/blob/4c28832d112060ec9944eafee745b403881e1daa/tf_pose/estimator.py#L388

    keypoints = np.empty([COCOKeypoints.Background.value, 3])
    for i in range(COCOKeypoints.Background.value):
        if i not in human.body_parts.keys():
            keypoints[i] = np.array([0.0, 0.0, 0.0])
        else:
            part = human.body_parts[i]
            x = part.x * image_width + 0.5
            y = part.y * image_height + 0.5
            confidence = part.score
            keypoints[i] = np.array([x, y, confidence])
    return keypoints
