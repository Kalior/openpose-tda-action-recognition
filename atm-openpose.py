from tracker import Tracker


if __name__ == '__main__':
    model_path = "/path/to/openpose/models/"

    tracker = Tracker(model_path)

    # print(dir(openpose))
    # image(openpose, "media/COCO_val2014_000000000192.jpg")
    tracker.video("media/video.avi")
