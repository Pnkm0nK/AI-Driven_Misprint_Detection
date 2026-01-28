import pdf2image
import os
import dotenv
pdf_dir_path = "./label_scans"
output_dir_path = "./OCR_bachelor/images"
dotenv.load_dotenv()
poppler_path = os.getenv("POPPLER_PATH")
for file_name in os.listdir(pdf_dir_path):
    if file_name.endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir_path, file_name)
        pdf2image.convert_from_path(
            pdf_path,
            dpi=300,
            fmt="jpeg",
            output_folder=output_dir_path,
            output_file=os.path.splitext(file_name)[0],
            poppler_path=poppler_path
            )