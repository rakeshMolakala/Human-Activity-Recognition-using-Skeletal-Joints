# -*- coding: UTF-8 -*-
# python main.py --ip 0.0.0.0 --port 8000

import cv2 as cv
import argparse
import numpy as np
import time
from utils import choose_run_mode, load_pretrain_model, set_video_writer
from Pose.pose_visualizer import TfPoseVisualizer
from Action.recognizer import load_action_premodel, framewise_recognize
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
# import socket
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
ip_address = s.getsockname()[0]
print(ip_address)
ap = argparse.ArgumentParser(description='Action Recognition by OpenPose')
ap.add_argument('--video', help='Path to video file.')
ap = argparse.ArgumentParser()
ap.add_argument('--video', help='Path to video file.')
ap.add_argument("-i", "--ip", type=str, required=True,
    help="ip address of the device")
ap.add_argument("-o", "--port", type=int, required=True,
    help="ephemeral port number of the server (1024 to 65535)")
ap.add_argument("-f", "--frame-count", type=int, default=32,
    help="# of frames used to construct the background model")
args = vars(ap.parse_args())
# args = parser.parse_args()

# 导入相关模型

#estimator = load_pretrain_model('VGG_origin')
# estimator = load_pretrain_model('mobilenet_thin')

# action_classifier = load_action_premodel('framewise_recognition.h2')

# # 参数初始化
# realtime_fps = '0.0000'
# start_time = time.time()
# fps_interval = 1
# fps_count = 0
# run_timer = 0
# frame_count = 0

# 读写视频文件（仅测试过webcam输入）

# video_writer = set_video_writer(cap, write_fps=int(7.0))


# # 保存关节数据的txt文件，用于训练过程(for training)
#f = open('e.txt', 'a+')
import paho.mqtt.client as mqtt
import requests
import json

def on_message(client, userdata, message):
    print("li")
    data1 =[]
    receivedstring = str(message.payload.decode("utf-8"))
    data1=receivedstring.split(",")
    print(data1)
    with open('config.json', 'w') as json_file:
        json.dump(data1, json_file)


broker_address="broker.hivemq.com"
client = mqtt.Client("SIEORA") 
client.connect(broker_address) 
client.on_message=on_message 
client.subscribe("HAR-DS")
# client.publish("HAR-AS","0")
# client2.subscribe("HAR-DS1")
response = 0

# broker_address="broker.hivemq.com"
# client2 = mqtt.Client("SIEORA") 
# client2.connect(broker_address) 
# client2.on_message=on_message2 
# client.subscribe("HAR-DS1")
with open('config.json') as f:
    data = json.load(f)
    # print(data[0])
 
serverToken = 'AAAAnshq6Rg:APA91bGWpYw7_95Ey2gC2zTZNahdaEWr3945hKUlshJMX0h7YiZlgOd3x7jgytCSfd2NjhsQLKD5JeCexTv1ylpcgM8Gofrk9CNwY2HvkRbNh8zzc7W0mOBeHkovEd30wPc_WC5Fwr7W'
deviceToken = data[0]
# deviceToken="eA1OkmuzzQs:APA91bHZDRC9DPWFERP3Yr4QJ6BVdFZsQ597XM2z5LUUzLHhZYy78inSHdN4h7vNnCwVfdNFBJQasT4AqEHjUnLcKveiLPsKheVuRykEkpeCWpkN4kUUxen8vOIFL_XX9iMaimiaVmIA"
headers = {
        'Content-Type': 'application/json',
        'Authorization': 'key=' + serverToken,
      }

body = {
          'notification': {'title': 'Sending push form python script',
                            'body': 'New Message'
                            },
          'to':
              deviceToken,
          'priority': 'high',
        #   'data': dataPayLoad,
        }
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")

def web_stream(frameCount):
    # grab global references to the video stream, output frame, and
    # lock variables
    global outputFrame, lock
    estimator = load_pretrain_model('mobilenet_thin')
    total = 0
    # action_classifier = load_action_premodel('framewise_recognition.h2')


    action_classifier = load_action_premodel('newharmodel2.model')
    # 参数初始化
    realtime_fps = '0.0000'
    start_time = time.time()
    fps_interval = 1
    fps_count = 0
    run_timer = 0
    frame_count = 0
    # global notification_flag
    notification_flag=0
    normalflag=0
    cap = cv.VideoCapture(0)
    # cap=choose_run_mode(args)
    while True:
        
        client.loop_start()
        # client2.loop_start()
        has_frame, show = cap.read()
        if has_frame:
            fps_count += 1
            frame_count += 1

            # pose estimation
            humans = estimator.inference(show)
            # get pose info
            pose = TfPoseVisualizer.draw_pose_rgb(show, humans)  # return frame, joints, bboxes, xcenter
            # recognize the action framewise
            show,label = framewise_recognize(pose, action_classifier)

            height, width = show.shape[:2]
            # 显示实时FPS值
            if (time.time() - start_time) > fps_interval:
                # 计算这个interval过程中的帧数，若interval为1秒，则为FPS
                realtime_fps = fps_count / (time.time() - start_time)
                fps_count = 0  # 帧数清零
                start_time = time.time()
            # fps_label = 'FPS:{0:.2f}'.format(realtime_fps)
            # cv.putText(show, fps_label, (width-160, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # 显示检测到的人数
            num_label = "Human: {0}".format(len(humans))
            cv.putText(show, num_label, (5, height-45), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # 显示目前的运行时长及总帧数
            if frame_count == 1:
                run_timer = time.time()
            run_time = time.time() - run_timer
            time_frame_label = '[Time:{0:.2f} | Frame:{1}]'.format(run_time, frame_count)
            # print(data1)
            if label == "falldown" and notification_flag ==0:
                normalflag=0
                if len(humans) == 1:
                    print(len(humans))
                    client.publish("HAR-AS","1"+","+str(ip_address))
                    serverToken = 'AAAAnshq6Rg:APA91bGWpYw7_95Ey2gC2zTZNahdaEWr3945hKUlshJMX0h7YiZlgOd3x7jgytCSfd2NjhsQLKD5JeCexTv1ylpcgM8Gofrk9CNwY2HvkRbNh8zzc7W0mOBeHkovEd30wPc_WC5Fwr7W'
                    deviceToken = data[0]
                    headers = {
                            'Content-Type': 'application/json',
                            'Authorization': 'key=' + serverToken,
                        }

                    body = {
                            'notification': {'title': 'Sending push form python script',
                                                'body': 'New Message'
                                                },
                            'to':
                                deviceToken,
                            'priority': 'high',
                            }
                    response = requests.post("https://fcm.googleapis.com/fcm/send",headers = headers, data=json.dumps(body))
                    print(response.status_code)
                    print(response.json())
                    notification_flag=1
            if label=='stand' or label=='standd' or label=='eat' or label == 'suicide':
                if normalflag==0:
                    client.publish("HAR-AS","0")
                    notification_flag=0
                    normalflag=1
                    
            # print(ip_address)
    

            cv.putText(show, time_frame_label, (5, height-15), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            cv.imshow('Action Recognition based on OpenPose', show)
            # video_writer.write(show)

            # # 采集数据，用于训练过程(for training)
            #joints_norm_per_frame = np.array(pose[-1]).astype(np.str)
            #f.write(' '.join(joints_norm_per_frame))
            #f.write('\n')
        if total > frameCount:
            total += 1
        with lock:
            outputFrame = show.copy()
        if cv.waitKey(1) & 0xFF == ord("q"):
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
            (flag, encodedImage) = cv.imencode(".jpg", outputFrame)

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
    app.run(host=args["ip"], port=args["port"],
        threaded=True, use_reloader=False)
   
# video_writer.release()
cap.release()
# f.close()
