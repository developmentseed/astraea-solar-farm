"""
Script demonstrating how to export a Keras model.

@author: Development Seed
"""

import os
import os.path as op
import sys
import shutil

import tensorflow as tf

#######################################
# Input processing functions
#######################################
HEIGHT, WIDTH, CHANNELS = 128, 128, 9


def serving_input_receiver_fn():
    """Convert b64 string encoded images into tensor"""

    # hardcoded normalization constants from Astraea's training
    means = tf.constant([1386.94997868, 1361.24726238, 1386.50767922, 2610.66932423,
           1566.00392023, 2327.5767376 , 2751.79511461, 2560.08313097,
           1698.32812001])
    std_devs = tf.constant([542.24409153,  614.51673764,  888.16536141,  858.40447766,
            796.98834181,  718.05002432,  884.17682263, 1094.57904506,
           1054.6599312])

    def decode_and_resize(image_str_tensor):
        """Decodes b64encoded array, resizes it and returns a uint16 tensor."""
        image = tf.decode_raw(tf.decode_base64(image_str_tensor), tf.uint16)
        return tf.reshape(image, [HEIGHT, WIDTH, CHANNELS])

    # Run processing for batch prediction.
    input_ph = tf.placeholder(tf.string, shape=[None], name='image_binary')
    images_tensor = tf.map_fn(
        decode_and_resize, input_ph, back_prop=False, dtype=tf.uint16)

    # Cast to float and run xception preprocessing on images (to scale [0, 65535]
    # to [-1, 1] using mean + std_dev)
    images_tensor = tf.cast(images_tensor, dtype=tf.float32)
    images_tensor = tf.divide(tf.subtract(images_tensor, means), std_devs)
    return tf.estimator.export.ServingInputReceiver(
        {'input_layer': images_tensor},
        {'image_bytes': input_ph})

if __name__ == "__main__":
    # Get a Keras model
    model = tf.keras.models.load_model(sys.argv[1])

    # Compile model (necessary for creating an estimator). However, no training will be done here
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_save_fpath = 'dummy_model.h5'
    model.save(model_save_fpath)

    tf_model = tf.keras.models.load_model(model_save_fpath)
    estimator = tf.keras.estimator.model_to_estimator(keras_model=tf_model,
                                                      model_dir='tmp')

    # TF adds a `keras` subdirectory (I'm not sure why), so need to copy the `checkpoint` file up one level
    files = os.listdir(op.join('tmp', 'keras'))
    for f in files:
        shutil.move(op.join('tmp', 'keras', f), op.join('tmp', f))
    estimator.export_savedmodel(
        'export',
        serving_input_receiver_fn=serving_input_receiver_fn)
