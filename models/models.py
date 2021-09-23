import tensorflow.keras as K
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten, Conv2D, GlobalAveragePooling2D, Dropout, BatchNormalization, Lambda
from tensorflow.keras.utils import get_file
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, ResNet101, EfficientNetB0, InceptionV3
import tensorflow_hub as hub
from vit_keras import vit

#data_normalization = Sequential(
#    [
#        layers.Rescaling(scale=1./127.5, offset=-1.),
#        layers.Resizing(224, 224),
#    ],
#    name="data_normalization",
#)


def vgg16(input_shape, num_classes, activation="softmax"):
    base_model = VGG16(
                    include_top=False,
                    weights="imagenet",  # Load weights pre-trained on ImageNet.
                    input_shape=input_shape, 
    )  # Do not include the ImageNet classifier at the top.
    # Freeze the base_model
    base_model.trainable = False
    
        #Build the model
    img_input = Input(shape=input_shape)
    #x = Lambda(lambda image: tf.image.resize(image, (224,224)))(img_input)
    #x = data_normalization(img_input)
    x = img_input
    x = K.applications.vgg16.preprocess_input(x)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    outputs = Dense(num_classes, activation=activation)(x) #Adds a fully connected layer with output = num_classes
    model = Model(img_input, outputs, name="vgg16")
    return model

def vgg19(input_shape, num_classes, activation="softmax"):
    base_model = VGG19(
                    include_top=False,
                    weights="imagenet",  # Load weights pre-trained on ImageNet.
                    input_shape=(224,224,3), 
    )  # Do not include the ImageNet classifier at the top.
    # Freeze the base_model
    base_model.trainable = False
    
        #Build the model
    img_input = Input(shape=input_shape)
    x = Lambda(lambda image: tf.image.resize(image, (224,224)))(img_input)
    #x = data_normalization(img_input)
    #x = img_input
    x = K.applications.vgg19.preprocess_input(x)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    #x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    #x = Dense(256, activation='relu')(x)
    #x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation=activation)(x) #Adds a fully connected layer with output = num_classes
    model = Model(img_input, outputs, name="vgg19")  
    return model

def resnet50(input_shape, num_classes, activation="softmax"):
    base_model = ResNet50(
                    include_top=False,
                    weights="imagenet",  # Load weights pre-trained on ImageNet.
                    input_shape=(224,224,3),  
    )  # Do not include the ImageNet classifier at the top.

    # Freeze the base_model
    base_model.trainable = False
    
    #Build the model
    img_input = Input(shape=input_shape)
    x = Lambda(lambda image: tf.image.resize(image, (224,224)))(img_input)
    #x = data_normalization(img_input)
    #x = img_input
    x = K.applications.resnet50.preprocess_input(x)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    #x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    #x = Dense(512, activation='relu')(x)
    #x = Dropout(0.3)(x)
    #x = BatchNormalization()(x)
    #x = Dense(256, activation='relu')(x)
    #x = Dropout(0.3)(x)
    #x = BatchNormalization()(x)
    #x = Dense(128, activation='relu')(x)
    #x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation=activation)(x) #Adds a fully connected layer with output = num_classes
    model = Model(img_input, outputs, name="resnet50")  
    return model

def resnet101(input_shape, num_classes, activation="softmax"):
    base_model = ResNet101(
                    include_top=False,
                    weights="imagenet",  # Load weights pre-trained on ImageNet.
                    input_shape=input_shape,  
    )  # Do not include the ImageNet classifier at the top.

    # Freeze the base_model
    base_model.trainable = False
    
    #Build the model
    img_input = Input(shape=input_shape)
    #x = Lambda(lambda image: tf.image.resize(image, (224,224)))(img_input)
    #x = data_normalization(img_input)
    x = img_input
    x = K.applications.resnet101.preprocess_input(x)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    #x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    #x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    #x = Dense(256, activation='relu')(x)
    #x = Dropout(0.3)(x)
    #x = BatchNormalization()(x)
    #x = Dense(128, activation='relu')(x)
    #x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation=activation)(x) #Adds a fully connected layer with output = num_classes
    model = Model(img_input, outputs, name="resnet101")  
    return model

def inception(input_shape, num_classes, activation="softmax"):
    base_model = InceptionV3(
                    include_top=False,
                    weights="imagenet",  # Load weights pre-trained on ImageNet.
                    input_shape=input_shape,  
    )  # Do not include the ImageNet classifier at the top.

    # Freeze the base_model
    base_model.trainable = False
    
    #Build the model
    img_input = Input(shape=input_shape)
    #x = Lambda(lambda image: tf.image.resize(image, (224,224)))(img_input)
    #x = data_normalization(img_input)
    x = img_input
    x = K.applications.inceptionv3.preprocess_input(x)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    #x = Dense(512, activation='relu')(x)
    #x = Dropout(0.3)(x)
    #x = BatchNormalization()(x)
    #x = Dense(256, activation='relu')(x)
    #x = Dropout(0.3)(x)
    #x = BatchNormalization()(x)
    #x = Dense(128, activation='relu')(x)
    #x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation=activation)(x) #Adds a fully connected layer with output = num_classes
    model = Model(img_input, outputs, name="inception")  
    return model

def efficientNetB0(input_shape, num_classes, activation="softmax"):
    base_model = EfficientNetB0(
                    include_top=False,
                    weights="imagenet",  # Load weights pre-trained on ImageNet.
                    input_shape=(224,224,3),
    )  # Do not include the ImageNet classifier at the top.
    # Freeze the base_model
    base_model.trainable = False
    
     #Build the model
    img_input = Input(shape=input_shape)
    x = Lambda(lambda image: tf.image.resize(image, (224,224)))(img_input)
    
    #x = data_normalization(img_input)
    #x = img_input
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    #x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    #x = Dense(512, activation='relu')(x)
    #x = Dropout(0.3)(x)
    #x = BatchNormalization()(x)
    #x = Dense(256, activation='relu')(x)
    #x = Dropout(0.3)(x)
    #x = BatchNormalization()(x)
    #x = Dense(128, activation='relu')(x)
    #x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation=activation)(x) #Adds a fully connected layer with output = num_classes
    model = Model(img_input, outputs, name="efficientnetb0")  
    return model

def get_hub_url_and_isize(model_name, ckpt_type, hub_type):
  if ckpt_type == '1k':
    ckpt_type = ''  # json doesn't support empty string
  else:
    ckpt_type = '-' + ckpt_type  # add '-' as prefix
  
  hub_url_map = {
    'efficientnetv2-b0': f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-b0/{hub_type}',
    'efficientnetv2-b1': f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-b1/{hub_type}',
    'efficientnetv2-b2': f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-b2/{hub_type}',
    'efficientnetv2-b3': f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-b3/{hub_type}',
    'efficientnetv2-s':  f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-s/{hub_type}',
    'efficientnetv2-m':  f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-m/{hub_type}',
    'efficientnetv2-l':  f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-l/{hub_type}',

    'efficientnetv2-b0-21k': f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-b0-21k/{hub_type}',
    'efficientnetv2-b1-21k': f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-b1-21k/{hub_type}',
    'efficientnetv2-b2-21k': f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-b2-21k/{hub_type}',
    'efficientnetv2-b3-21k': f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-b3-21k/{hub_type}',
    'efficientnetv2-s-21k':  f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-s-21k/{hub_type}',
    'efficientnetv2-m-21k':  f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-m-21k/{hub_type}',
    'efficientnetv2-l-21k':  f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-l-21k/{hub_type}',
    'efficientnetv2-xl-21k':  f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-xl-21k/{hub_type}',

    'efficientnetv2-b0-21k-ft1k': f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-b0-21k-ft1k/{hub_type}',
    'efficientnetv2-b1-21k-ft1k': f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-b1-21k-ft1k/{hub_type}',
    'efficientnetv2-b2-21k-ft1k': f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-b2-21k-ft1k/{hub_type}',
    'efficientnetv2-b3-21k-ft1k': f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-b3-21k-ft1k/{hub_type}',
    'efficientnetv2-s-21k-ft1k':  f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-s-21k-ft1k/{hub_type}',
    'efficientnetv2-m-21k-ft1k':  f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-m-21k-ft1k/{hub_type}',
    'efficientnetv2-l-21k-ft1k':  f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-l-21k-ft1k/{hub_type}',
    'efficientnetv2-xl-21k-ft1k':  f'gs://cloud-tpu-checkpoints/efficientnet/v2/hub/efficientnetv2-xl-21k-ft1k/{hub_type}',
      
    # efficientnetv1
    'efficientnet_b0': f'https://tfhub.dev/tensorflow/efficientnet/b0/{hub_type}/1',
    'efficientnet_b1': f'https://tfhub.dev/tensorflow/efficientnet/b1/{hub_type}/1',
    'efficientnet_b2': f'https://tfhub.dev/tensorflow/efficientnet/b2/{hub_type}/1',
    'efficientnet_b3': f'https://tfhub.dev/tensorflow/efficientnet/b3/{hub_type}/1',
    'efficientnet_b4': f'https://tfhub.dev/tensorflow/efficientnet/b4/{hub_type}/1',
    'efficientnet_b5': f'https://tfhub.dev/tensorflow/efficientnet/b5/{hub_type}/1',
    'efficientnet_b6': f'https://tfhub.dev/tensorflow/efficientnet/b6/{hub_type}/1',
    'efficientnet_b7': f'https://tfhub.dev/tensorflow/efficientnet/b7/{hub_type}/1',
  }
  
  image_size_map = {
    'efficientnetv2-b0': 224,
    'efficientnetv2-b1': 240,
    'efficientnetv2-b2': 260,
    'efficientnetv2-b3': 300,
    'efficientnetv2-s':  384,
    'efficientnetv2-m':  480,
    'efficientnetv2-l':  480,
    'efficientnetv2-xl':  512,
  
    'efficientnet_b0': 224,
    'efficientnet_b1': 240,
    'efficientnet_b2': 260,
    'efficientnet_b3': 300,
    'efficientnet_b4': 380,
    'efficientnet_b5': 456,
    'efficientnet_b6': 528,
    'efficientnet_b7': 600,
  }
  
  hub_url = hub_url_map.get(model_name + ckpt_type)
  image_size = image_size_map.get(model_name, 224)
  return hub_url, image_size

def efficientNetV2(input_shape, num_classes, activation="softmax"):
    #base_model = effnetv2_model.get_model('efficientnetv2-b0', include_top=False)

    model_name = 'efficientnetv2-b0' 
    ckpt_type = '1k'   
    hub_type = 'feature-vector' 
    hub_url, image_size = get_hub_url_and_isize(model_name, ckpt_type, hub_type)

    base_model = hub.KerasLayer(hub_url, trainable=False, name="efficientNetV2")
    rescale=1./255

    #Build the model
    img_input = Input(shape=input_shape)
    x = Lambda(lambda image: tf.image.resize(image, (image_size,image_size)) * rescale -1)(img_input)
    #x = data_normalization(img_input)
    x = base_model(x, training=False)
    #x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    #x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    #x = Dense(512, activation='relu')(x)
    #x = Dropout(0.3)(x)
    #x = BatchNormalization()(x)
    #x = Dense(256, activation='relu')(x)
    #x = Dropout(0.3)(x)
    #x = BatchNormalization()(x)
    #x = Dense(128, activation='relu')(x)
    #x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation=activation)(x) #Adds a fully connected layer with output = num_classes
    model = Model(img_input, outputs, name="efficientnetv2")  
    return model


def ViT(input_shape, num_classes, activation="softmax"):
    base_model = vit.vit_b32(
        image_size = 224,
        activation = 'softmax',
        pretrained = True,
        include_top = False,
        pretrained_top = False,
        classes=num_classes)

    # Freeze the base_model
    base_model.trainable = False

    #Build the model
    img_input = Input(shape=input_shape)
    x = Lambda(lambda image: tf.image.resize(image, (224,224)))(img_input)
    #x = img_input
    #x = vit.preprocess_inputs(img_input)
    x = vit.preprocess_inputs(x)
    x = base_model(x, training=False)
    x = Flatten()(x)
    #x = GlobalAveragePooling2D()(x)
    #x = BatchNormalization()(x)
    #x = Dense(512, activation='relu')(x)
    #x = Dropout(0.2)(x)
    #x = BatchNormalization()(x)
    #x = Dense(512, activation='relu')(x)
    #x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    #x = Dense(128, activation='relu')(x)
    #x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation=activation)(x) #Adds a fully connected layer with output = num_classes
    model = Model(img_input, outputs, name="vit-b32")  
    return model


def leNet(input_shape, num_classes, activation="softmax"):   
    
    inputs = Input(input_shape)
    c1 = Conv2D(16, (5, 5), activation='relu', padding='same') (inputs)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.1) (c1)
    c1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (5, 5), activation='relu', padding='same') (c1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.1) (c2)
    c2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='relu', padding='same') (c2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.2) (c3)
    c3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='relu', padding='same') (c3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.2) (c4)
    c4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation='relu', padding='same') (c4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.3) (c5)
    c5 = MaxPooling2D(pool_size=(2, 2)) (c5)
    
    f = Flatten()(c5)

    # Fully Connected 1
    d1 = Dense(1000, activation='relu')(f)
    d1 = BatchNormalization()(d1)
    d1 = Dropout(0.3) (d1)
    # Fully Connected 2
    d2 = Dense(100, activation='relu')(d1)
    d2 = BatchNormalization()(d2)
    d2 = Dropout(0.3) (d2)
    # Fully Connected 3
    outputs = Dense(num_classes, activation=activation)(d2)
    
    model = Model(inputs=[inputs], outputs=[outputs], name="LeNet")
    return model  


if __name__ == '__main__':
    leNet((32, 32, 3), 10).summary()

    
    