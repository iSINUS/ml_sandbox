#!/bin/bash

docker image prune -f
docker-compose up --build -d