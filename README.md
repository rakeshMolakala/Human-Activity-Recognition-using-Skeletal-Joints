# Human-Activity-Recognition-using-Skeletal-Joints# Safety Monitoring System for Dementia Patients Using Deep Learning Approach

## Description

Dementia is one of the most common disorders observed in the senior population nowadays. The most common cause is Alzheimer's disease. And keeping an eye on and caring for these people has been crucial. This initiative focuses on HAR since it is critical to track the movements of the elderly and individuals in coma in unsupervised circumstances. This study demonstrates a clever human action recognition approach that uses skeletal joint kinematics to automatically recognize human actions and integrate competencies. 

This is a low-cost, high-accuracy approach. When people are alone, an independent smartphone application is being created to monitor their state and surroundings. The mobile application also includes a Notification API interface that allows for the transmission of alert notifications in the event of an abnormal condition. As a result, this project gives a proof of concept of a technique that can be used to assist old citizens and youngsters in the event of disasters or health difficulties.

## Objectives
* The main goal of the system is to use human skeletal joints position to determine Human Activity. 
* To design an intelligent human action and human gesture recognition system which can process the live input images or video and develop a model based on that to identify abnormal human actions.
* Here we are going to capture the different skeletal images of human actions and collect it as data sets, train the data based on algorithms using deep learning.
* For live streaming we will make an application using react native framework for detecting any abnormal activity in the house.
* An alert notification will also appear in our mobile if an abnormal activity is detected.

## Dependencies
pip install flask == 1.1.1
pip install sklearn == 0.23.2
pip install tensorflow == 2.3.1
pip install tinydb == 3.15.2
pip install face_recognition
pip install imutils
pip install opencv == 4.2.0
pip install socket
pip install numpy == 1.18.5
pip install flask
pip install requests
pip install threading
pip install json

## Executing Program
```
run the code -- python main.py --ip 0.0.0.0 --port 8000
```

## System Architecture

<img width="466" alt="HARFlow" src="https://user-images.githubusercontent.com/91920989/186406552-31a20178-1018-49a0-8de2-6dd4751a233e.png">

## Modules

* Human activity collection
    * We captured the live data from the standard user and recorded certain poses which resemble the human activity in real life like fall, sleep, sit, stand. 
    * Each of these pose’s folders are separately made which contains frames captured from the live camera. 
    * The most critical task in this project is preparing a dataset, as we have prepared our own dataset through live data, we used deep learning technique to recognize action to expect unique outcomes than traditional ones. 
    * <img width="413" alt="harpic2" src="https://user-images.githubusercontent.com/91920989/186424411-706590cd-fcb4-4c76-8792-a58077342412.png">

* Labelling of dataset
    * Data labelling is necessary for a number of applications, including computer vision and natural language processing. It also needed when in the domain of speech recognition.  
    * After the live data is collected, the next process is labelling which is done by tagging each input with certain pose as sit, stand, fall down, etc. for a certain frame. 
    * The trained model's accuracy is dependent on the labelled dataset's accuracy, hence taking the time and resources to achieve highly accurate dataset labelling is critical.
    * <img width="429" alt="harpic3" src="https://user-images.githubusercontent.com/91920989/186427520-6dc6367d-e4f5-49f2-8fa1-7140e298a09a.png">
    
* Developing a model file
    * The sequential algorithm of CNN network is utilized to provide maximum accuracy. Here it is used to train the dataset consisting of human activity. 
    * The sequential methodology is well suited in the scenario as there is a simple stack of layers with one input tensor layer and one output one for each layer taken. 
    * Another advantage is, in Keras the simplest technique to build a model is sequential one. It allows you to construct the model layer and layer and process the input in such a way, not as a bulk one at a time. 
    * Each layer is associated with the weights that equals to the one above it. 
    * Add() function is used to add two layers and one output layer.
    * Dense layer is a common type of layer that is well utilized in this situation, which enables to connect all the nodes in the previous layer to the nodes in the current layer.
    * ![harpic4](https://user-images.githubusercontent.com/91920989/186431448-30df91b0-e64d-4565-9e74-55ddc70f73b2.png)
    
* Live streaming
    * The use cases of IOT(Internet of things) are vast in the present world where everything is on automation process. 
    * In our case, using IOT to display the output recognition and transferring signals is well suited as the system needs a well-placed camera to capture actions. 
    * We live stream the camera capturing data like video and audio and will display it to the end user. 
    * This process of transmitting or getting media in this manner is referred to as streaming.
    * The phrase “LIVE” refers to as the medium of the data delivery, rather than the medium itself, it’s a replacement of a file downloading, where the end user have to download the complete file before listening or watching it.
    * ![harpic5](https://user-images.githubusercontent.com/91920989/186436325-1f23e452-e059-4d15-b004-f52a6a430e4f.jpg)
    * ![harpic6](https://user-images.githubusercontent.com/91920989/186436713-acf6a2c6-f146-4f0a-ab39-12f4c2558306.jpg)


* React Native Application
    * In the system, the mobile app development acts as the interface that end user can access the system. 
    * A React Native framework is used in the case to create a hierarchy of user interface components.
    * When an abnormal circumstance happens during the live streaming, an alert message is sent to the user as a warning.
    * ![harpic7](https://user-images.githubusercontent.com/91920989/186437237-6563ad98-fc35-4eff-90d7-27c059c59956.jpg)
    * ![harpic8](https://user-images.githubusercontent.com/91920989/186437442-70086122-b140-4468-978d-b53524b50ff3.jpg)
    * ![harpic9](https://user-images.githubusercontent.com/91920989/186437573-f77cab85-57b2-4cc9-b542-27a66686d7ae.jpg)

## Advantages of the Proposed System
* Accuracy increases with the use of advanced algorithms
* Avoids mishaps with elderly as well as children
* Live monitoring using independent mobile app developed
* Saves the life of people

## Authors
@RakeshMolakala
@NikhilObili

## Citations
* Learning Complex Spatio-Temporal Conﬁgurations of Body Joints for Online Activity Recognition Jin Qi, Zhangjing Wang, Xiancheng Lin, and Chunming Li, IEEE 2018, Vol No: 2168-2291.
* A Hierarchical Spatio-Temporal Model for Human Activity Recognition Wanru Xu, Zhenjiang Miao, Member, IEEE, Xiaoping zhang, Senior Member, IEEE, Yi Tian, vol:1520-9210, 2017.
* Murad, Abdulmajid, and Jae-Young Pyun. "Deep recurrent neural networks for human activity recognition." Sensors 17.11 (2017): 2556.
* Ensemble Manifold Rank Preserving for Acceleration-Based Human Activity Recognition Dapeng Tao, Lianwen Jin, Member, IEEE, Yuan Yuan, Senior Member, IEEE, and Yang Xue, vol: 2162-237X, June 2016. 

