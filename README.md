# astraea-solar-farm

Instructions and a script for compiling Astraea's solar farm detection classifier into a [TensorFlow Serving](https://github.com/tensorflow/serving) image.

To export a model as a TensorFlow estimator, run:
```sh
python receiver_fn.py [keras_model_path]
```

This will use the custom `serving_input_receiver_fn` to create an estimator in the `exports` folder. To run the docker container with this model:

```
docker run -p 8501:8501 -v $PWD/exports:/models/export -e MODEL_NAME=export -t tensorflow/serving:nightly
```

The above will connect to an exported model on disk. You can also save the model to the container if you want to upload the docker image to docker hub. See the example here: https://www.tensorflow.org/serving/docker. Copying from that example, you can see the steps needed to create a standalone image:

```sh
docker run -d --name serving_base tensorflow/serving
docker cp models/<my model> serving_base:/models/<my model>
docker commit --change "ENV MODEL_NAME <my model>" serving_base <org>/<image_name>:<version_tag>
docker kill serving_base
docker run -p 8501:8501 -t <org>/<image_name>:<version tag>
```

An example with filled in data:
```sh
docker run -d --name serving_base tensorflow/serving
docker cp solar_farm serving_base:/models/solar_farm
docker commit --change "ENV MODEL_NAME solar_farm" serving_base developmentseed/astraea-solar-farm:0.1-cpu
docker kill serving_base
docker run -p 8501:8501 -t developmentseed/astraea-solar-farm:0.1-cpu
```

For GPU images, use the `tensorflow/serving:gpu` image, then run the other commands as written, finally running with:
```sh
docker run --runtime=nvidia -p 8501:8501 -t <org>/<image_name>:<version_tag w/ `-gpu`>
```
