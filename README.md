#Practicing using pre-trained models with MoveNet

This project uses MoveNet which is an ultra fast and accurate pose detection model. The original, pretrained MoveNet model is available at TF Hub.

Read more about it and/or download the model here:
https://www.tensorflow.org/hub/tutorials/movenet

Objective is to figure out when a salsa dancer completes a full rotation and then count the number of rotations CW and CCW in a time period. This could be used to quantify differences in dance styles (based on the amount of spins) and used to address potential muscular imbalance if a dancer only spins one direction. Additionally, it could be used as a practice tool for dancers trying do a certain number of spins in a practice session. A spin is quantified as a turn happening on one foot and takes about 2 counts to complete. The dancer must begin with their face towards the camera and then complete it by facing the camera again. Since salsa songs have bpm in range 140 and 250. That's around 4 beats per second. For every turn, we would want at least 4 snapshots. In the limit that a turn only takes one count, we would want a 4 snapshots for one beat (0.25 seconds). Therefore the maximum time resolution should be set to 1/16 seconds/frame or 16 fps. By default, iPhone cameras records video at 30 frames per second (fps) which is plenty. It is important to note that if the dancer is spotting, their face will rotate faster (by a factor of <2) than the rest of their body so 30 fps would be good minimum fps. Since we only want "spins", there should also be a maximum rotation time. For a 140 bpm song   

We want to locate landmarks and their relation to eachother. The front profile is easiest (most common) but we may also need to make sure that the side profile and back of the head are tracked because a spin will have multiple frames where the dancer is facing away. We don't want to count 180 spins; only >=360 spins. It would be great if there was a way for the algorithm to "remember" the trajectory of a certain landmark. Though this may not be neccessary if we I can figure out how to do this with only facial features and shoulders. 
           
Potential Complications include: 
- a dancer with super fast spotting
- a dancer with long hair
- various preps before the turn
- hair covering face
- Face going down/up

Plan: 
1) Use MoveNet to extract body features. Test it with different rotation angles. 
2) Now try image sequencing capabilities of MoveNet. Take a video spinning CW and CCW. The first thing to check is if the rotation direction can be found via the sequence. The turn direction would be mainly based off of the direction of movement of the facial features. 
3) The spin will only count if the body movements include FRONT FACING -> DIAGONAL CW/CCW -> SIDE CW/CCW -> BACK/NO FACE -> SIDE CCW/CW -> DIAGONAL CCW/CW ->	FRONT FACING
4) If the model were to not work for this purpose, I may need to apply transfer learning and retrain the a few layers of the model. 


