## Nural Network Compression with Application

ENNCA course at WUT in 2023

## Task1_0309

cifar_dataset.py, prepares an image classification dataset from a directory of images where each subdirectory represents a class. It also applies a series of transformations (color jitter, random horizontal flip, random rotation, random perspective) to augment the data and increases its variety.

In the net.py script, a Convolutional Neural Network (CNN) is defined using PyTorch's nn.Module. The network architecture comprises several convolutional layers, pooling layers, and fully connected layers. The CNN is then used to transform an image from the dataset to an output tensor of class probabilities.

## Task2_0317

from https://colab.research.google.com/drive/1At6amqGoWVGwOUg4XguBWmiZz4E2oKtg#scrollTo=m7B3-oh-EBsw notebook you need to finish basic training of the network with data augmentation and without, push your notebook and log history to the repository.
	
from ImageEmotionRecognition2023_A_student.ipynb you can find how face detector works on video frames, you should create a function to read the video example in the notebook, save all cropped faces from the detector you use to a folder, save the folder and your code in .py and push to repository.
	
from VideoAudioEmotionRecognition2023_A.ipynb you will find the pretrained CNN model, and example of LSTM and AttentionHead Unit, you should create a class VideoNet() that forward the cropped faces(converted to tensor), and output a 8 class classification vector. So expect Frames->CNN->deep features-> LSTM/AttentionHead->FC(linear layer, in LSTM case takes the last item, in AttentionHead takes cls token).

## Task3_0323

Task1: dataset create the dataset object that loads input and mask images, and another one loads input and one-hot-encoded mask tensors.
Task2:Build FFN-UP A,B,C block, generate some random tensors, make sure the forward pass ok.
Task3: Replace all convolution blocks with Inverted residual blocks.

## Task4_0413

Task1:
Build the dataset object, you should load both input image in LR and target image in HR. Prepare some visualization of the samples having some agumentation simulating blurring, noise and compress effects(You can do that from HR with some compression, downsampling)
Task2: 
Build Resnet encoder and bifpn decoder network using all branches at different resolutions. Use pixel shuffle for final output from decoder to get SR output matching the SR requirements, 3 times in this case. You should generate Y channel only inputs for this task, you should expect also Y channel outputs for your target. Make forward of the data, and upsample the Cb,Cr components 3 times, can convert the output back to RGB format once you got the output from the network.
Task3 : Collapsible linear blocks implementation Having the reference implementation in Keras, try to implement CLB in torch. Becareful of the conv2d parameters in keras mismatching in torch.

https://github.com/ARM-software/sesr/blob/master/models/model_utils.py You should test it with

ps: You only need to implement LinearBlock_c. And you can skip anything related to quant.


## Task5_0601

Task1: Split the 1000 samples into two groups, 50%,50%, one of them will be named calibration_dataset while the other will be validation_dataset. You should use calibration dataset to calibrate scale and zero point before performing actual quantization.
Task2: Make PTQ training for torchvision.models.quantization.mobilenet_v2, using PerchannelMinMax Quantization for weights and PerTensorMinMax Quantization separately.
Task3: Make Perchannel MovingAverageMinMax Quantization for weights and MovingAverage Pertensor Quantization for activation. Compare to results in 2., Check which one is better. Compare some scales and zero points of from the observers. Explain why one of the solution is better?

## Project

TASKS

1.Create the Dataset object that properly load the input image and target mask. Save the class definition in the git repo. display a few images without data augmentations, use some proper color map from matplotlib for better visualization. Remember for training, you need to preprocess the target mask with the maximum depth.

2.Use a pretrained efficient-b0 as backbone(encoder) and bifpn as neck(decoder). You need only to use features from resolution levels of 28x28, 14x14, 7x7, given inputs as 224x224. Which means that from decoder, you also need the features of resolution 28x28, 14x14, create 3 of such bifpn decoder blocks. Then you need to upsample the 28x28 and 14x14 features to 56x56, pass through another convolution layers to form 1 channel 56x56 data as final output, which means you should resize the target mask to 56x56 using bilinear interpolation.

Train with MSE loss function. You need 3 runs with different LR and 2 runs with data augmentation and without.

