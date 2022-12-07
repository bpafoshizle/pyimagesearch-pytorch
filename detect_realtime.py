# import the necessary packages
from torchvision.models import detection
from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import torch
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="frcnn-resnet",
    choices=["frcnn-resnet", "frcnn-mobilenet", "retinanet"],
    help="name of the object detection model")
ap.add_argument("-l", "--labels", type=str, default="coco_classes.pickle",
    help="path to file containing list of categories in COCO dataset")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# set a device we will be using to run the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the list of categories in the COCO dataset and then generate a 
# set of bounding box colors for each class
CLASSES = pickle.loads(open(args["labels"], "rb").read())
COLORS = np.random.uniform(0, 244, size=(len(CLASSES), 3))

# initialize a dictionary containing model name and its corresponding
# torchvision function call
MODELS = {
    "frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
    "frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
    "retinanet": detection.retinanet_resnet50_fpn
}

# load the model and set it to evaluation mode
model = MODELS[args["model"]](pretrained=True, progress=True,
    num_classes=len(CLASSES), pretrained_backbone=True).to(DEVICE)
model.eval()

# initialize the video stream, allow the camera to warm up,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = imutils.video.FileVideoStream("images/video.mp4").start()
time.sleep(2.0)
fps = FPS().start()

# Grab initial frame
frame = vs.read()

# loop over frames from the video stream
while frame is not None:
    # resize it to have a maximum width of 400 pixels
    frame = imutils.resize(frame, width=400)
    orig = frame.copy()

    # convert the from from BGR to RGB channel ordering and change
    # frame from channels last to channels first ordering
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.transpose((2, 0, 1))

    # add a batch dimension, scale the raw pixel intensities to the
    # range [0, 1], and convert the frame to a floating point tensor
    frame = np.expand_dims(frame, axis=0)
    frame = frame / 255.0
    frame = torch.FloatTensor(frame)

    # send the input to the device and pass it through the 
    # network to get the detections and predictions
    frame = frame.to(DEVICE)
    detections = model(frame)[0]

    # loop over the detections
    for i in range(0, len(detections["boxes"])):
        # extract the confidence (i.e. probability) associated with
        # the prediction
        confidence = detections["scores"][i]

        # filter out the weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extrace the index of the class label from the 
            # detections, then compute (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections["labels"][i])
            box = detections["boxes"][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")

            # draw the bounding box and label on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(orig, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(orig, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # show the output frame
    cv2.imshow("Frame", orig)
    key = cv2.waitKey(1) & 0xFF 

    # if the 'q' key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()
    
    # grab the frame from the threaded video stream
    frame = vs.read()

# stop the timer and display the FPS information
fps.stop()
print("[INFO] elapse time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()