.. _troubleshooting:

Troubleshooting
===============

When troubleshooting, set the environment variable :code:`VERBOSE=1` to get debug logs. 
Below are some common errors when using LeanDojo:

Installation
************

* I run into errors related to :code:`lxml` and :code:`grpcio` when running :code:`pip install lean-dojo` on a Mac with Apple silicon.

This is a known issue with these two packages on Apple silicon. You should install them using whatever way that works for you. See these `two <https://stackoverflow.com/questions/19548011/cannot-install-lxml-on-mac-os-x-10-9>`_ `posts <https://stackoverflow.com/questions/66640705/how-can-i-install-grpcio-on-an-apple-m1-silicon-laptop>`_ on Stack Overflow.

Tracing Repos
*************

* The process is killed when tracing a repo.

The most likely reason is your machine doesn't have enough memory. The amount of 
memory required depends on the repo you're tracing. For large repos, such as mathlib, you need at least 32 GB memory. If getting more memory is not an option, 
you can try a smaller repo. If your machine has enough memory but the process still gets killed, please check
whether your Docker has access to all resources of host machine (see `here <https://docs.docker.com/desktop/settings/mac/#resources>`_).

Interacting with Lean
*********************

* :code:`docker: Error response from daemon: invalid mount config for type "bind": bind source path does not exist`

Make sure Docker has access to the :code:`/tmp` directory. If you're using Docker Desktop, go to Settings -> Resources -> File sharing.
