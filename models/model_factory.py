from models.models import leNet, vgg19, vgg16, resnet101, resnet50, inception 
from models.models import efficientNetB0, efficientNetV2, ViT

def make_model(network, input_shape, num_classes):
    if network == 'LeNet':
        return leNet(input_shape, num_classes, activation="softmax")
    elif network == 'vgg19':
        return vgg19(input_shape, num_classes, activation="softmax")
    elif network == 'vgg16':
        return vgg16(input_shape, num_classes, activation="softmax")
    elif network == 'resnet50':
        return resnet50(input_shape, num_classes, activation="softmax")
    elif network == 'resnet101':
        return resnet101(input_shape, num_classes, activation="softmax")
    elif network == 'efficientnetb0':
        return efficientNetB0(input_shape, num_classes, activation="softmax")
    elif network == 'inception':
        return inception(input_shape, num_classes, activation="softmax")
    elif network == 'efficientnetv2':
        return efficientNetV2(input_shape, num_classes, activation="softmax")
    elif network == 'vit':
        return ViT(input_shape, num_classes, activation="softmax")
    else:
        raise ValueError('unknown network ' + network)