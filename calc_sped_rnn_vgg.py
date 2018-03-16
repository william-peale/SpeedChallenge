import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from keras import models
from keras.models import Model
from keras import layers
from keras.layers import Input,concatenate,Conv2D, Flatten, Dense, MaxPooling2D, Dropout, ZeroPadding2D, LSTM, Reshape, GRU, Concatenate, PReLU, Lambda, GaussianNoise
from keras.optimizers import adam, adagrad, rmsprop
from keras.applications import VGG16
from keras.callbacks import CSVLogger, Callback
from keras.regularizers import l2

class cool_back(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.xvals = []
        self.best_loss = 9999
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        self.valls = logs.get('val_loss')
        curr_loss = logs.get('loss')
        if(curr_loss > 25):
            curr_loss = 25
        self.losses.append(curr_loss)
        self.val_losses.append(self.valls)
        self.xvals.append(len(self.xvals))
        if(self.valls < self.best_loss):
            print("Loss went from: " + str(self.best_loss) + "to " + str(self.valls))
            self.best_loss = self.valls
            self.model.save_weights("best_loss.h5")
        plt.plot(self.xvals,self.losses, self.val_losses)
        plt.savefig("loss.png")
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return

image_size = 224

dropout_val = 0.5

DIM_ORDERING = "tf"
l2_val = 0.000

#Load the VGG model

image_size = 224
#Load the VGG model
input_1 = Input(shape=(image_size, image_size, 3))
input_2 = Input(shape=(image_size, image_size, 3))

ll1 = Lambda(lambda cool: cool/127.5 - 1)(input_1)
ll2 = Lambda(lambda cool: cool/127.5 - 1)(input_2)

g1 = GaussianNoise(0.04)(ll1)
g2 = GaussianNoise(0.04)(ll2)

vgg_conv_1_base = VGG16(weights='imagenet', include_top=False)
vgg_conv_2_base = VGG16(weights='imagenet', include_top=False)

#vgg_conv_2_base.get_layer(name='vgg16').name='vgg16_1'

#for i in range(15):
#    vgg_conv_1_base.layers.pop()

#for i in range(15):
#    vgg_conv_2_base.layers.pop()
    
for layer in vgg_conv_1_base.layers:
    layer.trainable = False
for layer in vgg_conv_2_base.layers:
    layer.trainable = False
    
for layer in vgg_conv_2_base.layers:
    layer.name += "_1"


x1 = vgg_conv_1_base(g1)
x2 = vgg_conv_1_base(g2)

x1 = MaxPooling2D((2,2))(x1)
x2 = MaxPooling2D((2,2))(x2)

rs1 = Reshape((1,4608))(x1)
rs2 = Reshape((1,4608))(x2)

c = concatenate([rs1,rs2],axis=1)

c = Dropout(0.5)(c)

r1 = LSTM(512,return_sequences=True,activation='linear', recurrent_dropout=0.5, kernel_regularizer=l2(l2_val))(c)

r1 = PReLU()(r1)

r1 = Dropout(0.5)(r1)

r2 = LSTM(512,activation='linear',recurrent_dropout=0.5, dropout=0.5, kernel_regularizer=l2(l2_val))(r1)

r2 = PReLU()(r2)

output = Dense(1)(r2)

model = Model(inputs=[input_1,input_2], outputs=[output])
model.summary()

x1 = np.load("x_first.npy")
x2 = np.load("x_second.npy")
y = np.load("y.npy")

model.compile(optimizer=adagrad(lr=0.0008), loss='mse', metrics=["mean_squared_error"])

csv_logger = cool_back()

model.fit([x1,x2], [y], epochs=200, batch_size=32, validation_split=0.2,callbacks=[csv_logger])

model.save_weights("weights.h5")
