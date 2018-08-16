import numpy as np

# Tensorflow implementation of openpose:
try:
    from tf_pose.estimator import TfPoseEstimator
    from tf_pose.networks import get_graph_path, model_wh
except ImportError:
    print("Tensorflow openpose not available.")

from ..util import COCOKeypoints
# ^ Should be the same as tf_pose.common.CocoPart
# from tf_pose import common


class TFOpenpose:
    """A tensorflow implemtentation of OpenPose.

    Requires https://github.com/ildoonet/tf-pose-estimation to be installed
    on the system.  Currently not part of the Dockerfiles.

    """

    def __init__(self):
        self.tf_openpose = TfPoseEstimator(get_graph_path("mobilenet_thin"),
                                           target_size=(432, 368))

    def detect(self, original_image):
        """Detects the pose of every person in the given image.

        Parameters
        ----------
        original_image : image, to predict people of.

        Returns
        -------
        keypoints : array-like
            Contains the keypoints of every identified person in the image,
            shape = [n_people, n_keypoints, 3]
        image_with_keypoins : image, the original image overlayed with the
            identified keypoints.

        """
        image_height, image_width = original_image.shape[:2]

        humans = self.tf_openpose.inference(original_image, resize_to_default=True)
        image_with_keypoints = TfPoseEstimator.draw_humans(
            original_image, humans, imgcopy=True)
        people = np.array([self._tf_openpose_human_to_np(human, image_width, image_height)
                           for human in humans])
        return people, image_with_keypoints

    def _tf_openpose_human_to_np(human, image_width, image_height):
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
