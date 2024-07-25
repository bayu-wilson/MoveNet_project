### Motivation
The objective of this computer vision project is to track the number of times a single dancer spins/turns in each direction (over right of left shoulder). This project could be expanded on and applied to live fitness and movement applications. For example, a professional dancer may want to keep track of movements like spinning to quantify their training routine. This can help to ensure a dancer sufficiently practices to improve but does not over-train, risking injury. Tracking number of spins in both directions is important as well because it can help prevent muscular imbalances developed during biased training. It can also monitor creativity biases during improvised dancing or performances (i.e. a choreographer spinning in one direction significantly more than the other). Lastly, it could be used in an ethnomusicological context. On page 55, [table 1.1](https://drive.google.com/file/d/13SPHCf94HYKWoEIaKNVrBEtmyad39JaZ/view?usp=drive_link) in the book "Spinning Mambo into Salsa" by Juliet McMains, one of the major aesthetic differences between Palladium-Era Mambo of the 1950's and the Modern New York-Style Salsa/Mambo today is the number of spins/turns done by the dancers. Palladium-Era dancers typically would do few simple turns and idiosyncratic solo moves while modern dancers today incorporate multiple, complex turns. Computer vision is a tool worth considering in the context of quantitative dance ethnography.


### Methodology and Tools
In this project I am taking advantage of a pre-trained pose detection model made by Google Research, called **[MoveNet](https://t.co/QpfnVL0YYI?amp=1)**. Copied from [TF Hub](https://www.tensorflow.org/hub/tutorials/movenet), it is an "*ultra fast and accurate model that detects 17 keypoints of a body. The model is offered on [TF Hub](https://tfhub.dev/s?q=movenet) with two variants, known as Lightning and Thunder. Lightning is intended for latency-critical applications, while Thunder is intended for applications that require high accuracy. Both models run faster than real time (30+ FPS) on most modern desktops, laptops, and phones, which proves crucial for live fitness, health, and wellness applications.*". I found this model on TensorFlow hub, which is a useful library containing ample resusable machine learning components like this. This helps to build ML applications without starting from scratch.

#### How MoveNet works
The prediction scheme loosely follows [CenterNet](https://arxiv.org/abs/1904.07850) which models an object as a single point, the center point of its bounding box. This approach is faster than bounding box based detectors. I recall learning about bounding box based detectors in the Deep Learning Specialization I took from Andrew Ng. In these types of algorithms, an object detecting method is first used to tightly encompass the object this may use sliding windows or a complex arrangement of overlapping bounding anchor boxes similar to the YOLO algorithm. Note: Anchor boxes are used to help grid cells detect multiple overlapping objects, Non-max suppression is a way to make sure algorithms detects each object only once, and IoU is intersection over unions which is a measure of overlap. 

The CenterNet/MoveNet is simpler and more efficient alternative. It ends up being a standard keypoint estimation problem. Keypoints are just notable features/landmarks in an image/video such as body/face parts. The images are fed into a fully convolutional network (FCN) (this just means CNN's with no fully connected dense layers at the end). So the output could be a spatial map instead of a single vector of class scores. In this case, the FCN is used to generate a heatmap where the object centers are the peaks. Then the main idea is that this enables a rich feature map to be output where keypoints are able to be identified easily. 

The model was trained using the COCO standar benchmark dataset as well as an internal Google dataset called Active. It was important to augment the COCO dataset with more challenging poses and motion blur which is common in fitness and dance applications.

One of the most significant speed ups was to use lower resolution inputs to the models. To make up for the loss in resolution, a more intelligent cropping method is used which allowed the model to devote more of its attention and resources to the main subject, and not the background.


### Defining a turn
Though this project can be easily generalized to other genres of dance, I will set the contraints based off of salsa dancing. I will define salsa music to have a BPM ranging from 140 and 250 (2.3 and 4.2 beats per second respectively). A spin ber a beat in a 250 BPM song is difficult so we can define that as the fastest spin. In order to classify the spin, we would require at least 4 snapshots during movement. Therefore 4.2 spins per second * 4 snapshots = **16.4 FPS as the minimum frame rate**. Luckily my iPhone has FPS of 30 which is plenty.  

We also define a minimum time in which a spin must be completed. In salsa the slowest type of turn takes around 3 counts/beats. In a 140 BPM song, that is 3 beats / 2.3 BPS = 1.3 seconds. It turns out it is easier to deal with number of frames so 1.3 seconds * 30 fps = 39 frames. **A turn must occur within 1.3 seconds of 39 frames**

### Quantifying a turn
In order to quantify a turn, I first define 4 orientations: facing front, right, back, and left. To simplify the definition of a full turn, the starting and ending point of the turn must be towards the camera (facing front). A right turn would mean an orientation pattern of FRONT->RIGHT->BACK->LEFT->FRONT and a left turn would mean FRONT->LEFT->BACK->RIGHT->FRONT. 

I classify orientation simply using only 3 keypoints: the left shoulder, the right shoulder, and the nose. If nose is further to the left, then the person is left facing and vice versa to the right. And then the shoulder orientation can be easily used to define forward vs backward facing.



```
def get_orientation(x_ls,x_rs,x_n):
    """
    x_ls: x-coordinate of left shoulder
    x_rs: x-coordinate of right shoulder
    x_n: x-coordinate of nose
    """
    if (x_rs<x_n)&(x_n<x_ls):
        return "front"
    elif (x_rs<x_n)&(x_ls<x_n):
        return "left"
    elif (x_ls<x_n)&(x_n<x_rs):
        return "back"
    elif (x_n<x_ls)&(x_n<x_rs):
        return "right"
    else:
        return "Error: something went wrong."
```




### Limitations of this work as well as future work that could be done
- This has only been tested on a single dancer. In principle, it should also work with multiple dancers.
- One could try transfer learning and remove the last few layers of the fully-convolutional network and then retrain it with video data of partner dancing. Relative keypoints in the context of partner dancing could be interesting and novel. 
- It's possible that an orientation pattern like FRONT->RIGHT->BACK->RIGHT->BACK->LEFT->FRONT could be done within the time constraint. My spin tracking method may not count this. 
- This hasn't been tested on noisy data yet.
- There are definitely improvements that could be made to make this code more efficient.
