# Install openpose globally where all other python packages are installed.
from openpose import openpose as op


class CaffeOpenpose(object):
    """CMU's original OpenPose installation.

    Requires OpenPose to be installed on the system, there is a Dockerfile
    that achieves this, and there are installatino instructions on CMU's
    OpenPose GitHub page.

    Parameters
    ----------
    model_path : str
        Path to where the downloaded model to use is located,
        part of the installation procedure of OpenPose.

    """

    def __init__(self, model_path):
        # Initialise openpose
        params = self._openpose_parameters(model_path)
        # Construct OpenPose object allocates GPU memory
        self.openpose = op.OpenPose(params)

    def _openpose_parameters(self, model_path):
        params = {
            "logging_level": 3,
            "output_resolution": "-1x-1",
            "net_resolution": "-1x192",
            "model_pose": "COCO",
            "alpha_pose": 0.6,
            "scale_gap": 0.3,
            "scale_number": 1,
            "render_threshold": 0.05,
            # If GPU version is built, and multiple GPUs are available, set the ID here
            "num_gpu_start": 0,
            "disable_blending": False,
            # Ensure you point to the correct path where models are located
            "default_model_folder": model_path
        }
        return params

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
        keypoints, image_with_keypoints = self.openpose.forward(original_image, True)
        return keypoints, image_with_keypoints
