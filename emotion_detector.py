# USAGE
# python3 emotion_detector.py --cascade haarcascade_frontalface_default.xml --model output/epoch_75.hdf5 --ip 0.0.0.0 --port 8000

# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from flask import Response
from flask import Flask

from flask import render_template
import threading
import datetime
import time
import numpy as np
import argparse
import imutils
import cv2
import urllib.request
import urllib.parse
import requests
import json
from ubidots import ApiClient
api = ApiClient(token="A1E-Y0FHvT8NFn29e0Cv9O5vN9JlkVYvQj")


variable1 = api.get_variable("5d3daca7c03f9752cfa4d75b")
variable2 = api.get_variable("5d550d04c03f9708d88d0db7") 
variable3 = api.get_variable("5d550cf5c03f9708bbf50f1d")
variable4 = api.get_variable("5d550ed0c03f9709ddb82928")

send_url = "http://api.ipstack.com/103.81.238.70?access_key=ad7c97b207925f101642719b8ccc9f78"
geo_req = requests.get(send_url)
geo_json = json.loads(geo_req.text)
latitude = geo_json['latitude']
longitude = geo_json['longitude']
variable2.save_value({'value':longitude})
variable3.save_value({'value':latitude})
variable4.save_value({"value":10,"context":{"lat":latitude,"lng":longitude}})
city = geo_json['city']

url='http://192.168.1.3:8080/shot.jpg'
url1='http://192.168.1.17:8080/shot.jpg'
apikey='q9wYxbxY1nQ-gvEThnIMhy0Or9Eok0rf2cZADoA6Or'
numbers=('9360269504')
message=('someone is attacking' "  " + str(longitude) + " " +str(latitude))
sender='TXTLCL'
username = 'yadav.wegot@gmail.com'
send_msg=0
def sendSMS(apikey, numbers, sender, message):
    data =  urllib.parse.urlencode({'username':username,'apikey': apikey, 'numbers': numbers,
        'message' : message, 'sender': sender})
    data = data.encode('utf-8')
    request = urllib.request.Request("https://api.textlocal.in/send/?/")
    f = urllib.request.urlopen(request, data)
    fr = f.read()
    return(fr)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
	help="path to where the face cascade resides")
ap.add_argument("-m", "--model", required=True,
	help="path to where the model file resides")
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-i", "--ip", type=str, required=True,
	help="ip address of the device")
ap.add_argument("-o", "--port", type=int, required=True,
	help="ephemeral port number of the server (1024 to 65535)")
ap.add_argument("-f", "--frame-count", type=int, default=32,
	help="# of frames used to construct the background model")
args = vars(ap.parse_args())

outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

# load the face detector cascade, emotion detection CNN, then define
# the list of emotion labels
detector = cv2.CascadeClassifier(args["cascade"])
model = load_model(args["model"])

EMOTIONS = ["angry", "scared", "happy", "sad", "surprised",
	"neutral"]

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
	camera = cv2.VideoCapture(0)

# otherwise, load the video
else:
	camera = cv2.VideoCapture(args["video"])

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

def web_stream(frameCount):
	# grab global references to the video stream, output frame, and
	# lock variables
	global vs, outputFrame, lock, send_msg
	total = 0

	# keep looping
	while True:
		# grab the current frame
		(grabbed, frame) = camera.read()

		# if we are viewing a video and we did not grab a
		# frame, then we have reached the end of the video
		if args.get("video") and not grabbed:
			break

		# resize the frame and convert it to grayscale
		frame = imutils.resize(frame, width=300)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# initialize the canvas for the visualization, then clone
		# the frame so we can draw on it
		canvas = np.zeros((220, 300, 3), dtype="uint8")
		frameClone = frame.copy()

		# detect faces in the input frame, then clone the frame so that
		# we can draw on it
		rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
			minNeighbors=5, minSize=(30, 30),
			flags=cv2.CASCADE_SCALE_IMAGE)

		# ensure at least one face was found before continuing
		if len(rects) > 0:
			# determine the largest face area
			rect = sorted(rects, reverse=True,
				key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
			(fX, fY, fW, fH) = rect
			# extract the face ROI from the image, then pre-process
			# it for the network
			roi = gray[fY:fY + fH, fX:fX + fW]
			roi = cv2.resize(roi, (48, 48))
			roi = roi.astype("float") / 255.0
			roi = img_to_array(roi)
			roi = np.expand_dims(roi, axis=0)

			# make a prediction on the ROI, then lookup the class
			# label
			preds = model.predict(roi)[0]
			label = EMOTIONS[preds.argmax()]
			if label == "angry" and send_msg == 0:

				resp =  sendSMS(apikey, numbers,sender, message)
				send_msg = 1
				print("send msg",message)
				
			# draw the label on the frame
			cv2.putText(frameClone, label, (fX, fY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
			cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
				(0, 0, 255), 2)

			# loop over the labels + probabilities and draw them
			for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):

				
				# construct the label text
				text = "{}: {:.2f}%".format(emotion, prob * 100)


				# draw the label + probability bar on the canvas
				w = int(prob * 300)
				cv2.rectangle(canvas, (5, (i * 35) + 5),
					(w, (i * 35) + 35), (0, 0, 255), -1)
				cv2.putText(canvas, text, (10, (i * 35) + 23),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45,
					(255, 255, 255), 2)
				
			
			
		# show our classifications + probabilities
		cv2.imshow("Face", frameClone)
		cv2.imshow("Probabilities", canvas)
		#frame = vs.read()
		frame = imutils.resize(frame, width=400)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (7, 7), 0)

		# grab the current timestamp and draw it on the frame
		timestamp = datetime.datetime.now()
		cv2.putText(frame, timestamp.strftime(
			"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

		# if the total number of frames has reached a sufficient
		# number to construct a reasonable background model, then
		# continue to process the frame
		if total > frameCount:
			total += 1
		# acquire the lock, set the output frame, and release the
		# lock
		with lock:
			outputFrame = frame.copy()
		

		# if the 'q' key is pressed, stop the loop
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break
def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments

	# start a thread that will perform motion detection
	t = threading.Thread(target=web_stream, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()

	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)

# release the video stream pointer
#vs.stop()

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
