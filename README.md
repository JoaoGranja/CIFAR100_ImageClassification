# CIFAR100_ImageClassification

## **Project Description**
In this project, a step by step process will be developed for the Image Classification task over the [CIFAR100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). Image classification is the process of a model classifying input images into their respective category classes.  In this Image Classification task, I will use several pre-trained models with known architectures like VGG, Resnet, EfficientNet and recently ViT. Based on this pre-trained models, I will apply the transfer learning and fine-tuning techniques in order to achive high accuracy.
I will compare the performance of all models looking on loss and accuracy metrics over the testing dataset.

## **Dataset**
[CIFAR100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) will be used which is comprised of 60000 32x32 color images in 100 classes, with 600 images per class. This is a  challenge Image Classification task where it is important to use deep learning model to achieve good results. Actually, the small image resolution and the small number of images per class make it a difficult task to train from scratch a model.

## **Project Steps**
The [image_classification_main.ipynb](https://github.com/JoaoGranja/CIFAR100_ImageClassification/blob/master/image_classification_main.ipynb) script was created to train and evaluate the deep learning models over the [CIFAR100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). In this project, I took the following steps:

<ul>
  <li><strong>Colab preparation</strong> - In this step,  all necessary packages/libraries are installed and my google drive account is shared.</li>
  <li><strong>Configuration and Imports</strong> - All modules are imported and the 'args' dictionary is built with some configuration parameters. </li>
  <li><strong>Loading the dataset</strong> - CIFAR100 dataset is loaded and the data are analysed. </li>
  <li><strong>Data pre-processing and data augmentation</strong> - Apply one hot enconding for the output class and some data augmentation is used. </li>
  <li><strong>Optimizer</strong> - Choose the optimizer for model training </li>
  <li><strong>Model</strong> - Based on 'args' configuration, make the model. The model architecture is built on models module. </li>
  <li><strong>Training</strong> - The training process runs in this step. Several callbacks are used to improve the trainig process. </li>
  <li><strong>Visualize models result</strong> - After the model is trained, the accuracy and loss of the model is plotted.</li>
  <li><strong>Evaluation</strong> - After all models are trained, the evaluation over a testing dataset is done. </li>
</ul>

## **Conclusion**
Comparing the results obtained from all 6 models VGG19, Resnet50, efficientNetB0, efficientnetv2, vit-b32 and vit_scratch, it is possible to conclude that the model with the best result is vit-b32 followed by Resnet50. Indeed vit-b32 has the highest accuracy evaluated on testing dataset, around 91% which is very impressive. The second best model, Resnet50 achieved 78%.
It is important to note that all pre-trained models achieve good results, higher than 72%. Probably tuning hyperparameters of the models and training more epochs, that results could be even better. The unique model trained from scratch didn't achieve great results, around 36%, but there was a lot of space to improve that results. More epochs and use larger dataset were needed. 

## **Future Work**
The goal of this project was not to achieve the highest accuracy score over the [CIFAR100 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) but to approach the Image Classification task. For better results, several future works are usefull (use better data augmentation techniques, apply regularization techniques to reduce overfitting, tune hyperparameters of the model) 

