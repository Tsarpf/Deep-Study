# Deep-Study
Practice projects. Internet poker card classifier with a CNN in tensorflow (and pytorch) and RuneScape ore segmenter with OpenCV

## Deep card classifier

Augmented image examples | Original image examples
:----:|:----:
![](https://github.com/Tsarpf/Deep-Study/raw/master/pytorch%20practice/example_classifications_on_augmentations.png) | ![](https://github.com/Tsarpf/Deep-Study/raw/master/pytorch%20practice/example_validation_classifications.png)

### TensorFlow
TensorFlow version is a custom built CNN with layers selected and hyperparameters chosen by intuition and trial/error. 
Most of the work was put into image preprocessing and training set augmentation.

For some yet unknown reason it usually classifies all other cards successfully except it misclassifies 3c as 6c or vice versa.

Some additional difficulty arises from the fact that the card images are from two completely different poker sites, where for example the card with the value ten, is written T in the other and 10 in the other. But both are classified correctly as T.

The net reaches approx 99.4-99.7% validation set accuracy:
![](https://github.com/Tsarpf/Deep-Study/raw/master/tensorflow%20practice/loss%20figures.png)

### PyTorch
The PyTorch version is transfer-learned by retraining the last layer of 18 layers deep ResNet (resnet18), it only reaches around 80-90% classification accuracy. Probably unfreezing more than the last layer, and possibly having a shallower network might work, since the card images are rather small.


## RuneScape ore segmentation
Here it's running live, finding iron ore. It runs at least 30fps on my desktop.
![](https://github.com/Tsarpf/Deep-Study/raw/master/ore-classifier/runescape.gif)

While I initially planned to use deep learning for this, the OpenCV version (that I was originally building just for creating a training set for the deep learning algorithm) was already so good I didn't think there was any point in doing a deep learning version. Got some nice pictures ouf of the OpenCV experiments though:

RGB | HSV
:----:|:----:
![](https://github.com/Tsarpf/Deep-Study/raw/master/ore-classifier/results/rgb%20scatter%20small.png) | The lower right of the HSV graph nicely shows the iron ore colors separated from the other colors: ![](https://github.com/Tsarpf/Deep-Study/raw/master/ore-classifier/results/hsv%20scatter%20small.png)

Only using HSV thresholding to segment the images leaves spots in the ores, and some random individual pixels are found here and there:

![](https://github.com/Tsarpf/Deep-Study/raw/master/ore-classifier/results/step_1_segment_by_color.png)

Using morphological erosion with a 2 by 2 kernel gets rid of the individual pixels:

![](https://github.com/Tsarpf/Deep-Study/raw/master/ore-classifier/results/step_3_erosion.png)

Using morphological closing after this, the spots in the ores are removed:

![](https://github.com/Tsarpf/Deep-Study/raw/master/ore-classifier/results/step_2_morph_closing.png)

After some experiments and trial & error with coal ore, finally adding one additional round of of erosion makes the segmenter work well  for both iron and coal ores, with the latter having more 1-2 pixel group noise, since there are a lot of black pixels like the ones found in coal ores in other objects in RuneScape.

Final result:
![](https://github.com/Tsarpf/Deep-Study/raw/master/ore-classifier/results/iron_finder.png)
