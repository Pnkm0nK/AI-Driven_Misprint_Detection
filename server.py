import fastapi
import json
import tempfile
from modules.LabelProcessor import LabelProcessor
from modules.LabelResult import LabelResult
from modules.ResultPostprocessor import ResultPostprocessor

label_processor = LabelProcessor()
app = fastapi.FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/inference")
async def latest_image(
    image: fastapi.UploadFile = fastapi.File(...), 
    gt_file: fastapi.UploadFile = fastapi.File(...)
):
    # 1. Read binary files into memory or disk
    image_bytes = await image.read()
    gt_bytes = await gt_file.read()
    
    # 2. Setup temporary files to interact safely with legacy file-path APIs
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[1]) as tmp_scan:
        tmp_scan.write(image_bytes)
        scan_path = tmp_scan.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_gt:
        tmp_gt.write(gt_bytes)
        gt_path = tmp_gt.name

    try:
        # 3. Execute the core vision pipeline
        # Process label using the saved temporary file path
        results: LabelResult = label_processor.process_label(scan=scan_path)
        
        # Instantiate postprocessor with the generated ground truth file path
        result_processor = ResultPostprocessor(results, gt_file_path=gt_path)
        
        # Trigger explicit summary metric calculations
        result_processor.generate_summary(
            summary_name=os.path.splitext(image.filename)[0], 
            output_dir=None, 
            verbose=True, 
            json_output=False
        )

        # 4. Construct response payloads
        response_data = {
            "template_type": results.template_type,
            "has_defect": result_processor.has_defect,
            "conclusion": result_processor.conclusion_text,
            "metrics": result_processor.metrics,
            "extracted_texts": result_processor.extracted_texts,
            "extracted_barcodes": result_processor.extracted_barcodes
        }
        
        return JSONResponse(status_code=200, content=response_data)

    finally:
        # 5. Clean up system disk resources reliably
        if os.path.exists(scan_path):
            os.remove(scan_path)
        if os.path.exists(gt_path):
            os.remove(gt_path)
