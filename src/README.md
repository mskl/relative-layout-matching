# Relative Layout Matching for Document Data Extraction
> Source code attachements for the thesis by Matyáš Skalický

## Annotation tool
To build and run the annotation tool, use:
```bash
docker-compose -f docker-compose.annotator.yml up --build
```

Once running, the annotation tool will be available at http://localhost:5005

## Training
The repository contains a docker-compose that runs both Jupyter Notebook environment for interactive development along with a TensorBoard service for experiment logging. Use the provided `Makefile` to easily interact with the interactive dev. environment:

```bash
# build the image and run interactive dev environment in the background
make run

# stop the interactive dev environment
make stop

# exec into the container that is running the jupyter notebooks
make bash

# obtain the logs from the jupyter notebook container
make logs
```

Individual trainings can be started from within the interactive dev environment mentioned above, or rather from standalone docker container instances. To run an empty docker-compose forever blocking the GPU3 run:
```bash
# start a training docker-compose with GPU3
docker-compose -f docker-compose.train.gpu.yml run -d -e NVIDIA_VISIBLE_DEVICES=3 --name "GPU3" jupyter sleep infinity

# exec into the gpu3 container
docker exec -it gpu3 bash

# inside the docker image, run the following to schedule a training
tsp python3 train.py --consistency 1 --triplet 1 --reconstruction 1 --optimizer adam --backbone resnet_unet_50 --batch-size 2 --epochs 70 --embdim 256 --dataset elections

# task spooler (tsp) allows to show scheduled jobs as well as their statuses/logs
tsp

# to tail the logs from the last training, run
tsp -t
```

Alternatively, a separate docker-compose can be created for each experiment:
```bash
docker-compose -f docker-compose.train.gpu.yml run -d -e NVIDIA_VISIBLE_DEVICES=3 jupyter python3 train.py --consistency 1 --triplet 1 --reconstruction 1 --optimizer adam --backbone resnet_unet_50 --batch-size 2 --epochs 70 --embdim 256 --dataset elections
```
