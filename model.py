import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Lambda, Flatten, Dense, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D, Cropping2D

lines = []
images = []
measurements = []

folder_path = 'data/'

with open(folder_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

"""
this block contains the addition of the data
including reading center , left and right images
and also fliping of all the images read before.
"""
correction_factor = 0.2
skip_flag = False
for line in lines:
    if not skip_flag:
        skip_flag = True
        continue
    for i in range(3):
        im_token = line[i].split('/')[-1]
        im = folder_path + 'IMG/' + im_token
        image = cv2.imread(im)
        images.append(image)
        images.append(cv2.flip(image, 1))

    m = float(line[3])
    measurements.append(m)
    measurements.append(-1.0 * m)
    measurements.append(m + correction_factor)
    measurements.append(-1.0 * (m + correction_factor))
    measurements.append(m - correction_factor)
    measurements.append(-1.0 * (m - correction_factor))

X_train = np.asarray(images)
Y_train = np.array(measurements)

print(X_train.shape, Y_train.shape)

"""
 using nvidia network from this link
 https://developer.nvidia.com/blog/deep-learning-self-driving-cars/
 we will be using only 4 covolution layers with maxpooling layer
 and one conv layer with dropout layer and 4 FC layers
 """

print("training .....")
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))

model.add(Cropping2D(cropping=((70, 24), (0, 0))))

# conv 1 with maxpooling 1
model.add(Conv2D(24, 5, activation="relu"))
model.add(MaxPooling2D(2, padding="same"))

# conv 2 with maxpooling 2
model.add(Conv2D(36, 5, activation="relu"))
model.add(MaxPooling2D(2, padding="same"))

# conv 3 with maxpooling 3
model.add(Conv2D(48, 5, activation="relu"))
model.add(MaxPooling2D(2, padding="same"))

# conv 4 with dropout
model.add(Conv2D(64, 3, activation="relu"))
model.add(Dropout(0.5))

# conv 5 with maxpooling 4
model.add(Conv2D(64, 3, activation="relu"))
model.add(MaxPooling2D(2, 1, padding="same"))

model.add(Flatten())

# FC 1
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))

# FC 2
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))

# FC 3
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))

# FC 4
model.add(Dense(1))

model.summary()

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, epochs=8)

model.save('model.h5')

# (48216, 160, 320, 3) (48216,)
# training .....
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# lambda_1 (Lambda)            (None, 160, 320, 3)       0
# _________________________________________________________________
# cropping2d_1 (Cropping2D)    (None, 66, 320, 3)        0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 62, 316, 24)       1824
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 31, 158, 24)       0
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 27, 154, 36)       21636
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 (None, 14, 77, 36)        0
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 10, 73, 48)        43248
# _________________________________________________________________
# max_pooling2d_3 (MaxPooling2 (None, 5, 37, 48)         0
# _________________________________________________________________
# conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 3, 35, 64)         0
# _________________________________________________________________
# conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928
# _________________________________________________________________
# max_pooling2d_4 (MaxPooling2 (None, 1, 33, 64)         0
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 2112)              0
# _________________________________________________________________
# dense_1 (Dense)              (None, 100)               211300
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 100)               0
# _________________________________________________________________
# dense_2 (Dense)              (None, 50)                5050
# _________________________________________________________________
# dropout_3 (Dropout)          (None, 50)                0
# _________________________________________________________________
# dense_3 (Dense)              (None, 10)                510
# _________________________________________________________________
# dropout_4 (Dropout)          (None, 10)                0
# _________________________________________________________________
# dense_4 (Dense)              (None, 1)                 11
# =================================================================
# Total params: 348,219
# Trainable params: 348,219
# Non-trainable params: 0
# _________________________________________________________________
# Train on 38572 samples, validate on 9644 samples
# Epoch 10/10
# 2020-09-08 06:35:15.304313: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
# 2020-09-08 06:35:15.332633: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
# 2020-09-08 06:35:15.332658: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
# 2020-09-08 06:35:15.332713: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
# 2020-09-08 06:35:15.332757: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
# 2020-09-08 06:35:15.570764: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:893] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2020-09-08 06:35:15.571643: I tensorflow/core/common_runtime/gpu/gpu_device.cc:955] Found device 0 with properties:
# name: Tesla K80
# major: 3 minor: 7 memoryClockRate (GHz) 0.8235
# pciBusID 0000:00:04.0
# Total memory: 11.17GiB
# Free memory: 11.10GiB
# 2020-09-08 06:35:15.571693: I tensorflow/core/common_runtime/gpu/gpu_device.cc:976] DMA: 0
# 2020-09-08 06:35:15.571716: I tensorflow/core/common_runtime/gpu/gpu_device.cc:986] 0:   Y
# 2020-09-08 06:35:15.571761: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1045] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0)
# 38572/38572 [==============================] - 94s 2ms/step - loss: 0.0285 - val_loss: 0.0224
# Epoch 2/10
# 38572/38572 [==============================] - 89s 2ms/step - loss: 0.0243 - val_loss: 0.0210
# Epoch 3/10
# 38572/38572 [==============================] - 90s 2ms/step - loss: 0.0232 - val_loss: 0.0217
# Epoch 4/10
# 38572/38572 [==============================] - 89s 2ms/step - loss: 0.0225 - val_loss: 0.0218
# Epoch 5/10
# 38572/38572 [==============================] - 90s 2ms/step - loss: 0.0219 - val_loss: 0.0202
# Epoch 6/10
# 38572/38572 [==============================] - 90s 2ms/step - loss: 0.0215 - val_loss: 0.0197
# Epoch 7/10
# 38572/38572 [==============================] - 90s 2ms/step - loss: 0.0214 - val_loss: 0.0207
# Epoch 8/10
# 38572/38572 [==============================] - 89s 2ms/step - loss: 0.0209 - val_loss: 0.0219
# Epoch 9/10
# 38572/38572 [==============================] - 90s 2ms/step - loss: 0.0210 - val_loss: 0.0202
