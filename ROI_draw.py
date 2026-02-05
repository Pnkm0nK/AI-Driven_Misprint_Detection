import os
import cv2
from ROIStorage import ROIStorage, ROICollection

def create_roi_gui(full_image: cv2.Mat | None,
                   roi_collection: ROICollection) -> ROICollection:

    assert full_image is not None, "Full label image is not loaded."
    
    # Zoom and pan state
    scale = 1.0
    min_scale = 0.1
    max_scale = 10.0
    pan_x = 0
    pan_y = 0

    roi_modes = ["text_regions", "barcode_regions"]  
    colors = [(0, 255, 0), (0, 0, 255)]  # Green for text, Blue for barcode
    current_mode_index = 0
    roi_coordinates = roi_collection[roi_modes[current_mode_index]] if roi_modes[current_mode_index] in roi_collection else {}

    clone = full_image.copy()
    drawing = False
    ix = iy = -1
    curr_x = curr_y = -1  # Track current mouse position
    pending = []  # list of (img_x0, img_y0, img_x1, img_y1) in image coordinates
    change_stack = []
    
    panning = False
    pan_start_x = pan_start_y = 0
    pan_x_start = pan_y_start = 0

    def get_display_image():
        """Generate the current display view with zoom and pan"""
        nonlocal scale, pan_x, pan_y
        
        h, w = clone.shape[:2]
        
        # Clamp pan to keep image visible
        max_pan_x = max(0, int(w * scale) - w)
        max_pan_y = max(0, int(h * scale) - h)
        pan_x = max(0, min(pan_x, max_pan_x))
        pan_y = max(0, min(pan_y, max_pan_y))
        
        # Resize image
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)
        scaled = cv2.resize(clone, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        
        # Crop for pan
        crop_x = int(pan_x)
        crop_y = int(pan_y)
        crop_w = min(w, scaled_w - crop_x)
        crop_h = min(h, scaled_h - crop_y)
        
        if crop_w <= 0 or crop_h <= 0:
            return clone.copy()
            
        cropped = scaled[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        
        # Pad if necessary
        if cropped.shape[0] < h or cropped.shape[1] < w:
            result = clone.copy()
            result[0:cropped.shape[0], 0:cropped.shape[1]] = cropped
            return result
        
        return cropped
    
    def display_to_image_coords(disp_x, disp_y):
        """Convert display coordinates to original image coordinates"""
        img_x = int((disp_x + pan_x) / scale)
        img_y = int((disp_y + pan_y) / scale)
        return img_x, img_y
    
    def image_to_display_coords(img_x, img_y):
        """Convert image coordinates to display coordinates"""
        disp_x = int(img_x * scale - pan_x)
        disp_y = int(img_y * scale - pan_y)
        return disp_x, disp_y

    def _mouse_cb(event, x, y, flags, param):
        nonlocal drawing, ix, iy, curr_x, curr_y, panning, pan_start_x, pan_start_y, pan_x_start, pan_y_start, scale, pan_x, pan_y
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            curr_x, curr_y = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                curr_x, curr_y = x, y  # Update current position while dragging
            elif panning:
                pan_x = pan_x_start - (x - pan_start_x)
                pan_y = pan_y_start - (y - pan_start_y)
        elif event == cv2.EVENT_LBUTTONUP:
            if drawing:
                drawing = False
                # Convert to image coordinates
                img_x0, img_y0 = display_to_image_coords(min(ix, x), min(iy, y))
                img_x1, img_y1 = display_to_image_coords(max(ix, x), max(iy, y))
                
                # Clamp to image bounds
                h, w = full_image.shape[:2]
                img_x0 = max(0, min(img_x0, w))
                img_y0 = max(0, min(img_y0, h))
                img_x1 = max(0, min(img_x1, w))
                img_y1 = max(0, min(img_y1, h))
                
                if img_x1 > img_x0 and img_y1 > img_y0:
                    pending.append((img_x0, img_y0, img_x1, img_y1))
        elif event == cv2.EVENT_RBUTTONDOWN:
            panning = True
            pan_start_x, pan_start_y = x, y
            pan_x_start, pan_y_start = pan_x, pan_y
        elif event == cv2.EVENT_RBUTTONUP:
            panning = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Zoom at mouse position
            old_scale = scale
            if flags > 0:  # Scroll up = zoom in
                scale = min(max_scale, scale * 1.1)
            else:  # Scroll down = zoom out
                scale = max(min_scale, scale / 1.1)
            
            # Adjust pan to zoom towards mouse cursor
            zoom_ratio = scale / old_scale
            pan_x = int(pan_x * zoom_ratio + x * (zoom_ratio - 1))
            pan_y = int(pan_y * zoom_ratio + y * (zoom_ratio - 1))

    window_name = "Create ROIs"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, _mouse_cb)

    print("Controls:")
    print("  Left-click-drag: Draw ROI")
    print("  Right-click-drag: Pan")
    print("  Mouse wheel: Zoom in/out")
    print("  +/-: Zoom in/out (keyboard)")
    print("  u: Undo last ROI")
    print("  1: Switch to text_regions mode")
    print("  2: Switch to barcode_regions mode")
    print("  q/ESC: Finish")

    while True:
        # Generate display with zoom/pan
        display = get_display_image()
        color = colors[current_mode_index]
        
        # Draw existing ROIs in display space
        for (x0, y0, x1, y1) in roi_coordinates.values():
            disp_x0, disp_y0 = image_to_display_coords(x0, y0)
            disp_x1, disp_y1 = image_to_display_coords(x1, y1)
            cv2.rectangle(display, (disp_x0, disp_y0), (disp_x1, disp_y1), color, 2)
        
        # Draw current rectangle being drawn
        if drawing and curr_x >= 0 and curr_y >= 0:
            cv2.rectangle(display, (ix, iy), (curr_x, curr_y), color, 2)
        
        # If there are pending ROIs, ask for a name
        if pending:
            x0, y0, x1, y1 = pending.pop(0)
            default_name = f"ROI_{len(roi_coordinates) + 1}"
            try:
                name = input(f"Enter name for ROI {default_name} (leave blank to use '{default_name}'): ").strip()
            except Exception:
                name = default_name
            if not name:
                name = default_name
            roi_coordinates[name] = (x0, y0, x1, y1)
            roi_collection[roi_modes[current_mode_index]] = roi_coordinates
            # Draw permanent rectangle on clone in image coordinates
            cv2.rectangle(clone, (x0, y0), (x1, y1), color, 2)
            print(f"Added ROI '{name}': (x={x0}, y={y0}, x2={x1}, y2={y1})")

        # Show zoom level
        cv2.putText(display, f"Zoom: {scale:.1f}x", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF
        
        if key == ord('u'):
            if roi_coordinates:
                last_roi_name = list(roi_coordinates.keys())[-1]
                del roi_coordinates[last_roi_name]
                roi_collection[roi_modes[current_mode_index]] = roi_coordinates
                clone = full_image.copy()
                print(f"Undid last ROI: {last_roi_name}")
        elif key == ord('1'):
            if current_mode_index != 0:
                clone = full_image.copy()
                current_mode_index = 0
                roi_coordinates = roi_collection[roi_modes[current_mode_index]] if roi_modes[current_mode_index] in roi_collection else {}
                print("Switched to text_regions mode.")
        elif key == ord('2'):
            if current_mode_index != 1:
                clone = full_image.copy()
                current_mode_index = 1
                roi_coordinates = roi_collection[roi_modes[current_mode_index]] if roi_modes[current_mode_index] in roi_collection else {}
                print("Switched to barcode_regions mode.")
        elif key == ord('q') or key == 27:
            break
        elif key == ord('+') or key == ord('='):
            scale = min(max_scale, scale * 1.2)
        elif key == ord('-') or key == ord('_'):
            scale = max(min_scale, scale / 1.2)

    cv2.destroyWindow(window_name)
    return roi_collection

if __name__ == "__main__":
    from ImageProcessor import ImageProcessor
    import dotenv
    dotenv.load_dotenv()
    PADDING = int(os.getenv("PADDING"))

    label_scan_pdf_path = "../label_scans/M333023W146.pdf"
    roi_json_path = "./roi_data/test.json"

    image_processor = ImageProcessor()
    full_label_image = image_processor.convert_pdf_to_image(label_scan_pdf_path)

    roi_storage = ROIStorage(full_label_image, roi_json_path=roi_json_path)

    start_x, start_y = roi_storage.establish_roi_starting_position(
        template_image_path="./images/logo_template.jpg", padding_x=PADDING)
    full_label_image = full_label_image[start_y:, start_x:]

    existing_rois = roi_storage.load_roi_json_data()

    updated_rois = create_roi_gui(full_label_image, existing_rois)
    roi_storage.save_roi_json_data(updated_rois)