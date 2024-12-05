from flask import Flask, render_template, Response, redirect, url_for, jsonify
import cv2
import numpy as np

app = Flask(__name__)
cap = None

# Load Tiny YOLO
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
classes = open("coco.names").read().strip().split("\n")

detected_objects = []

def gen_frames():
    global cap, detected_objects
    while True:
        if cap is None or not cap.isOpened():
            continue
        ret, frame = cap.read()
        if not ret:
            break

        height, width, channels = frame.shape

        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []
        detected_objects = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:  # Confidence threshold
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    detected_objects.append(classes[class_id])

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), font, 0.5, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/objects')
def objects():
    global detected_objects
    return jsonify(objects=detected_objects)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/start')
def start_video():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
    return redirect(url_for('index'))

@app.route('/stop')
def stop_video():
    global cap
    if cap is not None and cap.isOpened():
        cap.release()
        cap = None
    return "Video Stopped"

if __name__ == '__main__':
    app.run(debug=True)
