# sAIght

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Description

sAIght is a Python-based application designed to assist individuals with visual impairments. It leverages the power of Artificial Intelligence (AI) for object detection, providing real-time feedback about the surrounding environment.
sAight uses the camera feed from the user's device to continuously scan the environment. The AI model identifies and locates objects in the feed, and the application then provides audio feedback to the user based on these detections.


## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Object Detection**: sAIght uses advanced AI algorithms to identify and locate objects in the user's environment. This feature helps visually impaired users navigate their surroundings more effectively.

- **Audio Feedback**: The application provides audio feedback based on the detected objects, informing the user about their environment in a non-visual way.

## Installation

Follow these steps to set up your development environment:

1. **Set up a virtual environment**  
   Create a new virtual environment using the venv module to keep the dependencies required by this project separate from your global Python environment.

   ```bash
   python -m venv venv
    ```
2. **Activate the virtual environment**
    Activate the newly created virtual environment. This command may vary depending on your operating system.

    On Windows:

   ```bash
   venv\Scripts\activate
    ```
    On Unix or MacOS:
    ```bash
    source venv/bin/activate
    ```
3. **Install the dependencies**
    After activating the virtual environment, use pip to install the Ultralytics and the Supervision 0.3.0 libraries. 
    ```bash    
    pip install ultralytics
    ```
    ```bash    
    pip install supervision==0.3.0
    ```
    ```bash
    pip install pyttsx3
    ```

    *Para SpeechRecognition: pip install SpeechRecognition y pip install pyaudio
## Usage

To run the application, execute the `main.py` script from the command line:
    `python main.py`
## Contributing

Thank you for your interest in contributing to our project. However, at this moment, we are not actively seeking for new contributors. Please check back in the future as this may change. We appreciate your understanding and support.

## License

This project is licensed under the [MIT License](LICENSE.md).
