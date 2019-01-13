# Deep-Study
Practice projects. Internet poker card classifier with a CNN in tensorflow (and pytorch) and RuneScape ore segmenter with OpenCV

## Deep card classifier

Tensorflow version is a custom built CNN with layers selected and hyperparameters chosen by intuition and trial/error. 
Most of the work was put into image preprocessing and training set augmentation.

For some yet unknown reason it usually classifies all other cards successfully except it misclassifies 3c as 6c or vice versa.

Some additional difficulty arises from the fact that the card images are from two completely different poker sites, where for example the card with the value ten, is written T in the other and 10 in the other. But both are classified correctly as T.

The net reaches approx 99.4-99.7% validation set accuracy:
![](https://github.com/Tsarpf/Deep-Study/raw/master/tensorflow%20practice/loss%20figures.png)



