#!/bin/bash


if test $# -gt 0; then
	docker run -it -p 8080:8080 -v $(pwd):/app/ python:stock-ml $@
else
	docker run -it -p 8080:8080 -v $(pwd):/app/ python:stock-ml python train.py
fi
