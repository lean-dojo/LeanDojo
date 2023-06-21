.. _troubleshooting:

Troubleshooting
===============

Below are some common errors when using LeanDojo:


* The process is killed when tracing a repo.

The most likely reason is your machine doesn't have enough memory. The amount of 
memory required depends on the repo you're tracing. For large repos, such as recent 
versions of mathlib, you need at least 32 GB memory. If getting more memory is not an option, 
you can trace a smaller repo. If your machine has enough memory but the process still gets killed, please check
whether your Docker has access to all resources of host machine (see `here <https://docs.docker.com/desktop/settings/mac/#resources>`_).

