#! /usr/bin/sh
docker buildx build --platform linux/amd64,linux/arm64 -t yangky11/lean-dojo:latest --push - < docker/Dockerfile