# Image Component Mixer

Demonstrates the mixing of different images and their various components

## Features

* Opens multiple image formats
    * BMP
    * JPEG
    * DICOM
* Displays Metadata

## Preview

![Preview](preview.gif)

## How to use

Just download the release from gh releases and run the executable.

## How to run dev environment

1. Clone the repo
2. Create a virtual environment in the src directory

    ```bash
    cd <this repo>/src
    python -m venv ./ 
    ```

3. Install dependencies from requirements.txt

    ```bash
    pip install -r requirements.txt
    ```

4. Run the app

    ```bash
    python image_viewer.py 
    ```

