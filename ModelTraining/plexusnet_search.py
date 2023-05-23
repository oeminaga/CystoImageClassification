#%%
import os
from elegy import ModelBase
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
#%%
# Use scikit-learn to grid search the number of neurons
def create_model(depth=3,length=2, junction=3,initial_filter=8,ApplyTransformer=False,type_of_block="inception"):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= "3"
    from tensorflow.compat.v1.keras.backend import set_session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.compat.v1.Session(config=config))
    from plexusnet.architecture import PlexusNet,Configuration
    import tensorflow_addons as tfa
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.constraints import maxnorm
    # create model
    Configuration["num_heads"]=4
    model=PlexusNet(depth=depth,length=length,initial_filter=initial_filter,junction=junction,n_class=6, input_shape=(32,32), type_of_block=type_of_block,ApplyTransformer=ApplyTransformer,ApplyLayerNormalization=True, run_all_BN=False,GlobalPooling="avg").model
    opt=tf.optimizers.Adam()
    model.compile(optimizer=opt, metrics=["acc"], loss="categorical_crossentropy")   
    return model

#%%
train_ds=tf.keras.preprocessing.image_dataset_from_directory(
    "./Study",
    labels='inferred',
    label_mode='categorical',
    class_names=['BladderNormal', 'BladderWallAlteration', 'Diverticulum', 'Stones', 'Cancer', 'Inflammation'],
    color_mode='rgb',
    batch_size=16,
    image_size=(32, 32),
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="training",
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=True,
) 
X_train=[]
Y_train=[]
for x,y in train_ds:
    X_train.extend(np.array(x))
    Y_train.extend(np.array(y))

print(np.array(X_train).shape)
print(np.array(Y_train).shape)


#model=PlexusNet(depth=2,length=2,initial_filter=2,junction=2,n_class=6, input_shape=(32,32), type_of_block="inception",ApplyTransformer=False,ApplyLayerNormalization=True, run_all_BN=False,GlobalPooling="avg").model
#model.summary()

#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import GridSearchCV

model = KerasClassifier(model=create_model, epochs=1, batch_size=16, verbose=1, 
                        ApplyTransformer=False,
                        depth=3,
                        length=2,
                        junction=1,
                        type_of_block="inception",
                        initial_filter=4
                        )

depth = [3]
length= [2,3,4,5,6,7]
junction= [1,2,3]
ApplyTransformer=[False,True]
initial_filter=[4,8,16,24]
type_of_block=["inception", "resnet", "vgg", "soft_att"]
param_grid = dict(depth=depth,
                  length=length,
                  junction=junction,
                  ApplyTransformer=ApplyTransformer,
                type_of_block=type_of_block,
                  initial_filter=initial_filter
                 )
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=8, cv=2,verbose=100)
grid_result = grid.fit(X_train, Y_train)

import pandas as pd
pd.DataFrame(grid_result.cv_results_).sort_values("mean_test_score",ascending=False).to_csv("./search_result_plexusnet.csv")
print(grid_result.best_params_)
#{'ApplyTransformer': False, 'depth': 3, 'initial_filter': 8, 'junction': 1, 'length': 4, 'type_of_block': 'soft_att'}