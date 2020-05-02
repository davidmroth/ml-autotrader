#!/bin/bash


DOCKER_BIN=docker
DOCKER_ARGS="run -it -p 8081:8080 -v $(pwd):/app/ -v $(pwd)/data/tensorboard:/tmp/tensorboard python:stock-ml"

if test $# -gt 0; then
	$DOCKER_BIN $DOCKER_ARGS $@
else
	$DOCKER_BIN $DOCKER_ARGS python train.py
fi
