#===================================IMPORT===================================#
import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
import tensorflow_datasets as tfds


#===================================TRAINNING/TESTING-DATA===================================#
(train_ds, test_ds), ds_info = tfds.load('emnist/letters', 
    split=['train', 'test'], 
    shuffle_files=True, 
    as_supervised=True, 
    with_info=True
)

#===================================PREPROCESS-DATA===================================#
def preprocess_img(image, label):
    #Convert images from `uint8` -> `float32`.
    return tf.cast(image, tf.float32) / 255.0, label

#apply preprocess_img to both train and test dataset
train_ds = train_ds.map(preprocess_img)
train_ds = train_ds.batch(32)
test_ds = test_ds.map(preprocess_img)
test_ds = test_ds.batch(32)

#===================================MODEL===================================#
model = tf.keras.models.Sequential()

#===================================LAYERS-SETUP===================================#
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(27, activation='softmax'))

#===================================COMPILE===================================#
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_ds, epochs=10, validation_data=test_ds)

model.save('handwritten.model.keras')

#===================================LOAD-MODEL===================================#
model= tf.keras.models.load_model("handwritten.model.keras")


#===================================PREDICT===================================#
im_num = 1
while os.path.isfile(f"Letters/letter{im_num}.png"):
    try:
        img = cv2.imread(f"Letters/letter{im_num}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction= model.predict(img)
        print(f"This letter is {chr(ord('@')+np.argmax(prediction))}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        im_num+=1
