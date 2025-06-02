# DeepLabCut Image Inference and Cropping Pipeline

This repository contains a pipeline for processing images using DeepLabCut (DLC). The pipeline resizes images, performs inference using pre-trained DLC weights, and crops the images based on the resulting keypoints. This is particularly useful for isolating regions of interest within images, such as the belly region of newts for re-identification purposes.

## Description

The pipeline automates the process of:

1.  **Resizing Images:** Scales input images to a standardized size to ensure compatibility with the DLC model.
2.  **Performing Inference with DLC:** Utilizes a pre-trained DeepLabCut model to predict keypoints on the resized images.
3.  **Cropping Images:** Crops the original, unresized images based on the keypoint locations predicted by DLC. This allows for precise extraction of specific regions from the original high-resolution images.

This automated process streamlines the workflow for analyzing images with DeepLabCut, especially when the goal is to extract specific regions of interest based on DLC's keypoint predictions.

## Requirements

*   Python 3.11
*   The required Python packages are listed in `requirements.txt`.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd deeplabcut_images_inference_and_cropping
    ```

2.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

## Execution

To run the Streamlit application:

```bash
streamlit run .\src\streamlit_app.py
```

This will launch the application in your web browser, allowing you to upload images and process them through the pipeline.

## Script Descriptions

*   `src\belly_cropper.py`: Contains functions for cropping the images based on the keypoints predicted by DeepLabCut. It takes the original image and the keypoint coordinates as input and outputs the cropped image.
*   `src\inference.py`: Includes functions for performing inference using the DeepLabCut model. It loads the model, preprocesses the images, and predicts the keypoint locations.
*   `src\streamlit_app.py`: This script creates the Streamlit web application, providing a user interface for uploading images, running the inference and cropping pipeline, and displaying the results.

## Usage

1.  Configure the path with images through the Streamlit interface.
2.  The application will automatically resize them, perform inference using the DeepLabCut model, and crop the images based on the predicted keypoints.
3.  The original images and the cropped images will be displayed in the application.