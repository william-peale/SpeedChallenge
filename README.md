###Speed Challenge
	This is my solution(?) for the comma speed challenge. Full disclaimer here, I did not get this job as I goofed and set my validation split to 2% instead of 20% which 
messed things up. However… The model I used is still kinda cool and I thought it would be worth slapping on my github. I checked out some solutions to this on github, and most 
people who claim to have succeeded in the challenge use the optical flow method detailed in that one [medium post](https://chatbotslife.com/autonomous-vehicle-speed-estimation-from-dashboard-cam-ca96c24120e4) that everyone seemed to have copied. I have two problems with 
that method: It still uses partial CV, so you are not making full use of gradient descent and it's really just copying that medium post. Not naming names here, but I have looked 
at peoples’ solutions that are directly ripped from the code on that post. Same method, same architecture, even just plain old copy pasta code.  So I thought I’d be cool and do 
a fully end to end deep learning method to set myself apart from the rest. I went with a method similar to [this paper](https://arxiv.org/abs/1709.08429). While the model in 
that paper would have probably worked, I just can’t train something that complex on my sad computer. So I went with a pre-learned vgg network reshaped and shoved into a series 
of LSTM units. Surprisingly, this actually worked. My validation loss (after I fixed it) got down to about 7 with early stopping. This is not as great as the optical flow model, but considering it is using 
transfer learning and scaled down layer sizes for a sad computer this is really good. If I had more time I probably would have tried a fully trainable network or 3D 
convolutional units.

