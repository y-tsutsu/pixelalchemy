# pixelalchemy

A small playground project for experimenting with GPU programming using fragment shaders and compute shaders.  
This repository contains simple examples and demos to help understand shader-based processing, including basic image processing and GPGPU-style workflows.

## Setup

Install required system packages:

```console
$ sudo apt install -y libcairo2-dev libgirepository-2.0-dev
$ sudo apt install -y gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 gstreamer1.0-tools \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly
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

This will launch a demo that uses GStreamer together with a fragment shader to perform edge detection on video input.
