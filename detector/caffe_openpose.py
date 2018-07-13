# Install openpose globally where all other python packages are installed.
from openpose import openpose as op


class CaffeOpenpose(object):

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
        keypoints, image_with_keypoints = self.openpose.forward(original_image, True)
        return keypoints, image_with_keypoints
