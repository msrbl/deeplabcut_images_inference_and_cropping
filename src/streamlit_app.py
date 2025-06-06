import shutil
import streamlit as st

from config import LOG_FILE, APP_DIR, TEMP_DIR, RESULT_DIR, logger
from crop.belly_cropper import process_images_crop
from inference.inference import process_images
from utils import resize_images, group_images_by_id

def display_logs():
    try:
        with open(LOG_FILE, "r") as f:
            log_content = f.read()
            st.subheader("Logs:")
            st.text(log_content)
    except FileNotFoundError:
        return "Log file not found."

def main():
    st.title("DeepLabCut Image Processing")
    logger.info("Application started.")

    data_path = st.text_input("Enter the path to the data directory:")

    if data_path:
        
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
        if RESULT_DIR.exists():
            shutil.rmtree(RESULT_DIR)

        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        
        inference_path = APP_DIR / "inference"

        st.subheader("Step 1: Resizing Images")
        with st.spinner("Resizing images..."):
            resize_path = TEMP_DIR / "resized_images"
            resize_images(data_path, resize_path)
        st.success("Images have been resized successfully!")
        logger.info("Image resizing is completed.")
        
        st.subheader("Step 2: Processing Images with DeepLabCut")
        with st.spinner("Running DeepLabCut inference..."):
            label_dir = process_images(
                parent_path=inference_path,
                data_path=resize_path,
                output_path=TEMP_DIR / "labels",
            )
        st.success("DeepLabCut inference complete!")
        logger.info("DeepLabCut inference complete.")

        csv_path = next(label_dir.rglob("*.csv"))

        st.subheader("Step 3: Cropping Images")
        with st.spinner("Cropping images based on keypoints..."):
            process_images_crop(
                csv_path=csv_path,
                output_dir=RESULT_DIR,
            )
        st.success("Image cropping complete!")
        logger.info("Image cropping complete.")
        
        st.subheader("Step 4: Grouping Images by Triton ID")
        with st.spinner("Grouping images by last 4 digits..."):
            group_images_by_id(RESULT_DIR)
        st.success("Success!")
        logger.info("Image grouping complete.")

        st.subheader("Results")
        st.write(f"Resized images are located in: {TEMP_DIR / 'resized_images'}")
        st.write(f"Cropped images are located in: {RESULT_DIR}")

    display_logs()
    logger.info("Application finished.")
    
if __name__ == "__main__":
    main()