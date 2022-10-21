# Image Processing Toolbox

Browse and open DICOM, JPG, BMP files and view their metadata

## Features

* Opens multiple image formats
    * BMP
    * JPEG
    * DICOM
* Displays Metadata

## Preview

![Preview](preview.gif)

## How to install

Just download the release from gh releases, unzip then run the executable.

## How to run dev environment

1. Clone the repo
2. Create a virtual environment in the src directory

    ```bash
    cd <this repo>/src
    python -m venv ./
    ```

3. Activate the virtual environment

    ```bash
    Scripts\Activate.ps1 # for powershell
    ```

    ```bash
    Scripts\Activate.bat # for cmd
    ```

4. Install dependencies from requirements.txt

    ```bash
    pip install -r requirements.txt
    ```

5. Run the app

    ```bash
    python image_viewer.py 
    ```

## Architecture Block Diagram

![Architecture Block Diagram](resources/block_diagram.png)

