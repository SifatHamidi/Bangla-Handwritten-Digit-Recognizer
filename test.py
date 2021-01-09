# -*- coding: utf-8 -*-
"""Test

Here the final model are loaded and used to test the images to show its working accuracy.
"""

import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

def prepare(filepath):
    IMG_SIZE = 32
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype('float32')

model = tf.keras.models.load_model('/content/model_filter.h5')

test = ('/content/testing-all-corrected/testing-b/b00005.png')
image = prepare(test)
image = image/255

prediction = model.predict([image])
predicted_class_indices = np.argmax(prediction, axis = 1)
print(predicted_class_indices)

im = cv2.imread(test)
im_resized = cv2.resize(im, (256, 256), interpolation=cv2.INTER_LINEAR)

plt.imshow(cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB))
plt.show()
