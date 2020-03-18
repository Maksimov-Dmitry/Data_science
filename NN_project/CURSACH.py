from __future__ import print_function

import draw_dataset
import utils

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Activation
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt

#create_images(image dataset)-----------------------------------------------------------
count = 5
size = 5
images = draw_dataset.get_pictures(count, size)

#image_dataset_for_input
images_norm = np.array([(np.array(i).ravel()) for i in images])

first_video_class = np.array([0, 1, 2, 81, 27, 28, 29, 61, 14, 13, 12, 42,
                     17, 16, 15, 62])

second_video_class = np.array([3, 83, 36, 52, 69, 85, 53, 73, 56, 40, 87, 71, 
                      54, 38, 16, 66])
third_video_class = np.array([6, 7, 88, 56, 10, 39, 19, 67, 4, 84, 25, 26,
                     60, 44, 23, 87])
thorth_video_class = np.array([22, 21, 64, 80, 24, 37, 53, 85, 33, 65, 81, 49,
                     33, 80, 64, 21])

count_train = 8
count_test = 16 - count_train

x_train, y_train, x_test, y_test = utils.get_dataset_from_video(count_train, count_test, images_norm, 
                                                                first_video_class, second_video_class,
                                                                third_video_class, thorth_video_class)

y_train = to_categorical(y_train, len(images))
y_test = to_categorical(y_test, len(images))
     

#RNN create-------------------------------------------------------------------------------

model = Sequential()

model.add(SimpleRNN(128, input_shape=(x_train.shape[1:]), return_sequences=True))
model.add(Dense(len(images)))
model.add(Activation('softmax'))
model.summary() 

#opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

model.compile(loss='categorical_crossentropy', 
                 optimizer='rmsprop',
                 metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=4, batch_size=1, verbose=1, validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=1)

print('\nТочность на проверочных данных:', test_acc)

predictions = model.predict(x_test)

utils.save_images(predictions, images)

plt.plot(history.history['loss'])
plt.show()

for i in range(0, count_train * 4, 3):
    print("###################################################################")
    picture = x_test[i][0].reshape(count * size, count * size)
    draw_dataset.draw_picture(picture)   
    for j in range(0, 16):
        number = np.argmax(predictions[i][j])
        if number == np.argmax(y_test[i][j]):
            print("TRUE")
        else:
            print("FALSE")
        picture = images_norm[number].reshape(count * size,count * size)
        draw_dataset.draw_picture(picture)

for j in range (0, count_test):
    print("##########NEW_INPUT############################")
    num_rows = 4
    num_cols = 4
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    labels = np.array([np.argmax(i) for i in y_test[j]])
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        draw_dataset.plot_value_array(i, predictions[j], labels, len(images))
    plt.show()
