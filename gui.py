import tempfile
import os
import cv2
import streamlit as st
from concurrent.futures import ProcessPoolExecutor

# 1. This function runs inside the separate, isolated background process
def run_pipeline_isolated(scan_path, gt_path):
    # Heavy imports happen ONCE inside the process when the first task hits it
    from modules.LabelProcessor import LabelProcessor
    from modules.ResultPostprocessor import ResultPostprocessor
    
    # Using a global variable or custom function attribute to cache the model 
    # instance inside the persistent background worker process memory space
    if not hasattr(run_pipeline_isolated, "processor"):
        run_pipeline_isolated.processor = LabelProcessor()
        
    processor = run_pipeline_isolated.processor
    results = processor.process_label(scan=scan_path)
    
    anomaly_map = cv2.rotate(results.anomaly_map, cv2.ROTATE_90_COUNTERCLOCKWISE)
    postprocessor = ResultPostprocessor(results, gt_file_path=gt_path)
    
    postprocessor.generate_summary(summary_name="Isolated_Run", output_dir=None, json_output=False)
    
    mismatch_image = cv2.rotate(postprocessor.get_image_with_highlighted_mismatches(), cv2.ROTATE_90_COUNTERCLOCKWISE)
    aligned_image = cv2.rotate(results.aligned_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return aligned_image, mismatch_image, anomaly_map, postprocessor.summary_text, postprocessor.has_defect


# 2. Cache the Process Pool Executor so Streamlit doesn't kill it between button clicks
@st.cache_resource
def get_persistent_executor():
    # Keeps exactly ONE worker process alive in the background permanently
    return ProcessPoolExecutor(max_workers=1)


def main():
    st.set_page_config(page_title="Label defect detection", layout="wide")
    st.title("Label Defect Detection")

    executor = get_persistent_executor()

    scan_file = st.file_uploader("Upload scan file (Image/PDF)", type=["png", "jpg", "jpeg", "pdf"])
    gt_file = st.file_uploader("Upload ground truth file (JSON)", type=["json"])

    if st.button("Start Analysis"):
        if scan_file is None or gt_file is None:
            st.error("Please upload both the scan and ground truth files.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(scan_file.name)[1]) as tmp_scan:
                tmp_scan.write(scan_file.read())
                scan_path = tmp_scan.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_gt:
                tmp_gt.write(gt_file.read())
                gt_path = tmp_gt.name

            with st.spinner("Processing pipeline inside persistent worker..."):
                try:
                    # Submit the task to the already running background process
                    future = executor.submit(run_pipeline_isolated, scan_path, gt_path)
                    aligned_image, mismatch_image, anomaly_map, summary_text, has_defect = future.result()
                
                    if not has_defect:
                        st.success("No defects detected")
                    else:
                        st.warning("DEFECTS DETECTED! Assess the results below:")
                    
                    st.subheader("Aligned Image")
                    st.image(aligned_image, channels="BGR", use_container_width=True)

                    st.subheader("Information Mismatches")
                    st.image(mismatch_image, channels="BGR", use_container_width=True)
                    
                    st.subheader("Defect Map")
                    st.image(anomaly_map, channels="BGR", use_container_width=True)

                    st.subheader("Summary")
                    st.text(summary_text)

                except Exception as e:
                    st.error(f"Execution failed: {e}")

                finally:
                    if os.path.exists(scan_path):
                        os.remove(scan_path)
                    if os.path.exists(gt_path):
                        os.remove(gt_path)

# CRITICAL FOR WINDOWS: This protects the entry point when the process spawns
if __name__ == "__main__":
    main()