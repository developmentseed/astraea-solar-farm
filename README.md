# astraea-solar-farm

Instructions and a script for compiling Astraea's solar farm detection classifier into a [TensorFlow Serving](https://github.com/tensorflow/serving) image.

## Basic Serving

To export a model as a TensorFlow estimator, run:
```sh
python receiver_fn.py [keras_model_path]
```

This will use the custom `serving_input_receiver_fn` to create an estimator in the `exports` folder. To run the docker container with this model:

```
docker run -p 8501:8501 -v $PWD/exports:/models/export -e MODEL_NAME=export -t tensorflow/serving:nightly
```

## Standalone Docker Image

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

### Use

When the image is running, you can send it a series of base-64 encoded `numpy` arrays and it will return a float representing the classification score for solar farm detection.

E.g.
```python
import json
import requests
from base64 import b64encode

array # a numpy array with shape (128, 128, 9) and dtype('uint16'), representing the Digital Numbers of Bands 2, 3, 4, 8, 5, 6, 7, 11, 12 of Sentinel 2 L1C data

# encode this image and transform it into the required payload
b64_string = b64encode(array).decode('utf-8')
instances = []
instances.append({'image_bytes': b64_string })
payload = json.dumps({'instances': instances})

# send it to our serving image
r = requests.post('http://localhost:8501/v1/models/solar_farm:predict', data=payload)
content = json.loads(r.content)
content # { "predictions": [[0.785]] }
```
