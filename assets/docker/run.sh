#!/bin/bash

if test $# -gt 1; then
	docker run -it -v $(pwd):/app/ python:stock-ml python /app/basic_model.py
else
	docker run -it -v $(pwd):/app/ python:stock-ml $@ 
fi
