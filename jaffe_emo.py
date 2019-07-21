#%%
import os,cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras import backend as K
#K.set_image_dim_ordering('th')

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam

from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
import random

from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import keras
from keras.models import load_model

from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2


#%%
PATH = os.getcwd()
# Define data path
data_path = PATH + '/data'
data_dir_list = os.listdir(data_path)


img_rows=256
img_cols=256
num_channel=1

num_epoch=10

img_data_list=[]


for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
        #input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize=cv2.resize(input_img,(128,128))
        img_data_list.append(input_img_resize)
        
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data = img_data/255
img_data.shape
#%%

if num_channel==1:
	if K.image_dim_ordering()=='th':
		img_data= np.expand_dims(img_data, axis=1) 
		print (img_data.shape)
	else:
		img_data= np.expand_dims(img_data, axis=4) 
		print (img_data.shape)
		
else:
	if K.image_dim_ordering()=='th':
		img_data=np.rollaxis(img_data,3,1)
		print (img_data.shape)


#%%
        
num_classes= 7
num_of_samples= img_data.shape[0]
labels= np.ones((num_of_samples,),dtype='int64')  

labels[0:29]=0 #angry class
labels[30:59]=1 #disgust
labels[60:92]=2 #fear
labels[93:124]=3 #happy
labels[125:155]=4 #neutral
labels[156:187]=5 #sad
labels[188:]=6 #surprise  
 
names=['Angry','Disgust','fear','Happy','Neutral','Sad','Suprise']     
Y = np_utils.to_categorical(labels,num_classes)		

x,y= shuffle(img_data,Y, random_state=2)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=2)

i = 100
plt.imshow(X_train[i,0], interpolation='nearest')
print("label : ", y_train[i,:])

X_train.shape

        
       
#%%
#data_generator = ImageDataGenerator(
                       # featurewise_center=False,
                        #featurewise_std_normalization=False,
                       # rotation_range=10,
                        #width_shift_range=0.1,
                        #height_shift_range=0.1,
                        #zoom_range=.1,
                        #horizontal_flip=True)

#%%
batch_size=7 #batch size for training

nb_classes=7

nb_epoch=30

#img_rows, img_cols=50, 50

img_channels= 1

nb_filters=30

nb_pool=5

nb_conv=3

l2_regularization=0.01

input_shape= img_data[0].shape
#%%
                        
#regularization = l2(l2_regularization)                        

#img_input = Input(input_shape)
model = Sequential()


model.add(Conv2D(6, kernel_size=(5, 5),input_shape=input_shape,padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2),padding='same'))

model.add(Conv2D(16, kernel_size=(5, 5),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2),padding='same'))
#model.add(Dropout(0.5))

#model.add(Conv2D(32, kernel_size=(5, 5),padding='same'))
#model.add(Activation('relu'))
#model.add(MaxPooling2D((2, 2),padding='same'))
#model.add(Dropout(0.5))

model.add(Conv2D(120, kernel_size=(5, 5),padding='same'))
model.add(Activation('relu'))
#model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(84))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))



#%%

#model = Model(img_input, output)
#%%






#%%
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
#model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
#%%
model.summary()
#%%

fashion_train = model.fit(X_train, y_train, batch_size=batch_size,epochs=nb_epoch,verbose=1,validation_data=(X_test, y_test))

#%%
score = model.evaluate(X_test, y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])
#%%

model.save('jaffe_model.h5')

#%%

accuracy = fashion_train.history['acc']
val_accuracy = fashion_train.history['val_acc']
loss = fashion_train.history['loss']
val_loss = fashion_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
#%%
test_image = X_test[0:1]
def get_featuremaps(model, layer_idx, X_batch):
	get_activations = K.function([model.layers[0].input, K.learning_phase()],[model.layers[layer_idx].output,])
	activations = get_activations([X_batch,0])
	return activations

layer_num=3
filter_num=0

activations = get_featuremaps(model, int(layer_num),test_image)

print (np.shape(activations))
feature_maps = activations[0][0]      
print (np.shape(feature_maps))

if K.image_dim_ordering()=='th':
	feature_maps=np.rollaxis((np.rollaxis(feature_maps,2,0)),2,0)
print (feature_maps.shape)

fig=plt.figure(figsize=(16,16))
plt.imshow(feature_maps[:,:,filter_num],cmap='gray')
plt.savefig("featuremaps-layer-{}".format(layer_num) + "-filternum-{}".format(filter_num)+'.jpg')

num_of_featuremaps=feature_maps.shape[2]
fig=plt.figure(figsize=(16,16))	
plt.title("featuremaps-layer-{}".format(layer_num))
subplot_num=int(np.ceil(np.sqrt(num_of_featuremaps)))
for i in range(int(num_of_featuremaps)):
	ax = fig.add_subplot(subplot_num, subplot_num, i+1)
	#ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
	ax.imshow(feature_maps[:,:,i],cmap='gray')
	plt.xticks([])
	plt.yticks([])
	plt.tight_layout()
plt.show()
fig.savefig("featuremaps-layer-{}".format(layer_num) + '.jpg')
