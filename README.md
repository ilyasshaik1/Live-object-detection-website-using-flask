# Live-object-detection-website-using-flask
# Real-time Object Detection with YOLOv3-Tiny and Flask

This project demonstrates real-time object detection using the **YOLOv3-Tiny** model and **Flask** web framework. The application captures video from a webcam, processes each frame to detect objects using YOLO, and displays the video feed with bounding boxes around detected objects on a webpage. The list of detected objects is also displayed in real-time.

## Features
- Real-time video streaming from a webcam.
- Object detection using YOLOv3-Tiny model.
- Bounding boxes and class labels are drawn around detected objects in the video feed.
- Web interface to start/stop the video feed and display detected objects.

## Requirements

- Python 3.x
- Flask
- OpenCV (for computer vision tasks)
- NumPy

### Install Dependencies

To run the application, you need to install the necessary Python packages. You can do this by running the following commands:

pip install flask opencv-python numpy

Your project directory should look like this:

/your-project-folder
    /templates
        home.html
        index.html
    /static
    app.py
    yolov3-tiny.weights
    yolov3-tiny.cfg
    coco.names
Here are the full links to the YOLOv3-Tiny model files:

1. **YOLOv3-Tiny Weights**: [Download yolov3-tiny.weights](https://pjreddie.com/media/files/yolov3-tiny.weights)
2. **YOLOv3-Tiny Config**: [Download yolov3-tiny.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg)
3. **COCO Names**: [Download coco.names](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

You can download these files and place them in the appropriate directory in your project.

## Acknowledgments

This project uses the YOLOv3-Tiny model, which is implemented in the [Darknet repository](https://github.com/pjreddie/darknet) and licensed under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.html) license.

You can find the pre-trained YOLOv3-Tiny weights and configuration files in the [YOLO website](https://pjreddie.com/darknet/yolo/).
