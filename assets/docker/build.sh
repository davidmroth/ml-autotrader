#!/bin/bash

#docker build -t python:stock-ml .
#cat assets/docker/Dockerfile | docker build -t python:stock-ml -
docker build -f assets/docker/Dockerfile -t python:stock-ml assets/docker
