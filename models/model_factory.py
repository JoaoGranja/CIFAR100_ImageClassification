#from models.models import leNet, vgg19, vgg16, resnet101, resnet50, inception 
#from models.models import efficientNetB0, efficientNetV2 #, ViT
#from models.ViT import create_vit_classifier

def make_model(network, input_shape, num_classes):
    if network == 'LeNet':
        from models.models import leNet 
        return leNet(input_shape, num_classes, activation="softmax")
    elif network == 'vgg19':
        from models.models import vgg19
        return vgg19(input_shape, num_classes, activation="softmax")
    elif network == 'vgg16':
        from models.models import vgg16
        return vgg16(input_shape, num_classes, activation="softmax")
    elif network == 'resnet50':
        from models.models import resnet50
        return resnet50(input_shape, num_classes, activation="softmax")
    elif network == 'resnet101':
        from models.models import resnet101
        return resnet101(input_shape, num_classes, activation="softmax")
    elif network == 'efficientnetb0':
        from models.models import efficientNetB0
        return efficientNetB0(input_shape, num_classes, activation="softmax")
    elif network == 'inception':
        from models.models import inception
        return inception(input_shape, num_classes, activation="softmax")
    elif network == 'efficientnetv2':
        from models.models import efficientNetV2
        return efficientNetV2(input_shape, num_classes, activation="softmax")
    elif network == 'vit':
        from models.ViT import create_vit_classifier
        #return ViT(input_shape, num_classes, activation="softmax")
        return create_vit_classifier(input_shape, num_classes)
    else:
        raise ValueError('unknown network ' + network)