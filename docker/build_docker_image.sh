#! /usr/bin/sh
# This is for developers only. Users do not have to build Docker containers.
# To use LeanDojo with Docker, users just have to make sure Docker is installed and running.
docker buildx build --platform linux/amd64,linux/arm64 -t yangky11/lean-dojo:latest --push - < docker/Dockerfile
