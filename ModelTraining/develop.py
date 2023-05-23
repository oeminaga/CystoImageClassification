##############################################
#
#   This file is part of the project "ModelTraining".
#   (c) 2021 Okyaz Eminaga, Palo Alto, CA, USA  - MIT License
#   For licensing information, see LICENSE.md.
##############################################
#%%
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import tensorflow as tf
from tensorflow import layers
import numpy as np
import sklearn
import keras_cv
FOLDER_WITH_IMAGES_IN_CLASSES = "./Study"
classes_to_consider=[f for f in os.listdir(f"{FOLDER_WITH_IMAGES_IN_CLASSES}") if f!=".DS_Store"]
labels = []
for i,class_folder in enumerate(classes_to_consider):
    files = [f for f in os.listdir(f"{FOLDER_WITH_IMAGES_IN_CLASSES}/{class_folder}") if f!=".DS_Store"]
    labels.extend([i]*len(files))
print(classes_to_consider)

class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',
                                                 classes=np.unique(labels),
                                                 y=labels)

class_weight_keras = {}
for i, w in enumerate(class_weights):
    class_weight_keras[i]=w
print(class_weight_keras)
print(tf.__version__)
#%%

def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


def mix_up(ds_one, ds_two, alpha=0.2):
    # Unpack two datasets
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    batch_size = tf.shape(images_one)[0]

    # Sample lambda and reshape it to do the mixup
    l = sample_beta_distribution(batch_size, alpha, alpha)
    x_l = tf.reshape(l, (batch_size, 1, 1, 1))
    y_l = tf.reshape(l, (batch_size, 1))

    # Perform mixup on both images and labels by combining a pair of images/labels
    # (one from each dataset) into one image/label
    images = images_one * x_l + images_two * (1 - x_l)
    labels = labels_one * y_l + labels_two * (1 - y_l)
    return (images, labels)

train_ds=tf.keras.preprocessing.image_dataset_from_directory(
    FOLDER_WITH_IMAGES_IN_CLASSES,
    labels='inferred',
    label_mode='categorical',
    class_names=classes_to_consider,
    color_mode='rgb',
    batch_size=16,
    image_size=(224, 224),
    shuffle=True,
    seed=123,
    validation_split=0.05,
    subset="training",
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=True,
)

val_ds=tf.keras.utils.image_dataset_from_directory(
    FOLDER_WITH_IMAGES_IN_CLASSES,
    labels='inferred',
    label_mode='categorical',
    class_names=classes_to_consider,
    color_mode='rgb',
    batch_size=16,
    image_size=(224, 224),
    shuffle=True,
    seed=123,
    validation_split=0.05,
    subset="validation",
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=True,
)

# %%
AUTOTUNE = tf.data.AUTOTUNE

val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


rand_augment = keras_cv.layers.RandAugment(
    value_range=(0, 255),
    augmentations_per_image=3,
    magnitude=0.3,
    magnitude_stddev=0.15,
    rate=0.8,#9090909090909091,
)


# %%

data_augmentation = tf.keras.Sequential([
  layers.RandomZoom(-0.4,0.1)
])
def apply_rand_augment(inputs):
    inputs=data_augmentation(inputs)
    inputs= rand_augment(inputs)
    
    return inputs
# %%
train_ds = train_ds.cache().repeat(4).prefetch(buffer_size=AUTOTUNE)
train_ds = tf.data.Dataset.zip((train_ds,train_ds))

AUTO = tf.data.AUTOTUNE

train_ds_mu = train_ds.map(
    lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=0.2), num_parallel_calls=AUTO
)
aug_train_ds=train_ds_mu.map(lambda x, y: (apply_rand_augment(x), y))
modelbase=tf.keras.applications.MobileNetV3Small(include_top=False, classes=7)
# %%
y=layers.GlobalAveragePooling2D()(modelbase.output)
y=layers.Dense(7)(y)
y=layers.Activation("softmax")(y)
model=tf.keras.Model(modelbase.input, y)

# %%
model.summary()
# %%
lr_v=tf.keras.optimizers.schedules.CosineDecay(
    1e-4, 1000, alpha=0.0, name=None
)

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr_v,
    rho=0.9,
    momentum=0.9,
    epsilon=1e-07,), metrics=["acc"],loss='categorical_crossentropy')


my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=30),
    tf.keras.callbacks.ModelCheckpoint(filepath='./weights_mobilenet_v3/model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs_mobilenet_v3'),
]

model.fit(aug_train_ds, validation_data=val_ds,workers=4,epochs=50,class_weight=class_weight_keras,callbacks=my_callbacks)
