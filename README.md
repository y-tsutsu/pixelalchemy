# pixelalchemy

A small playground project for experimenting with GPU programming using OpenGL and Vulkan.
This repository contains simple examples and demos to help understand shader-based processing, including fragment shaders, compute shaders, basic image processing, and GPGPU-style workflows.

The main demo uses GStreamer for video input and can run the edge detection pipeline with either Vulkan or OpenGL/ModernGL.
The Vulkan path is the default backend, while the OpenGL path remains available for comparing the two APIs around the same video-processing task.

## Setup

Install required system packages:

```console
$ sudo apt install -y libcairo2-dev libgirepository-2.0-dev
$ sudo apt install -y gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 gstreamer1.0-tools \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly
$ sudo apt install -y glslang-tools vulkan-tools
```

Install pipenv:

```console
$ python -m pip install pipenv
```

Install Python dependencies:

```console
$ pipenv install --dev
```

## Run

Run the demo:

```console
$ pipenv run start
```

This launches the Vulkan version by default.

You can also choose the renderer explicitly:

```console
$ pipenv run start -- --backend vulkan
$ pipenv run start -- --backend opengl
```

Both backends use GStreamer video input and perform edge detection on video frames.
