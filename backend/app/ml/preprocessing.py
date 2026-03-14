
import cv2
import numpy as np

def preprocess_for_ocr(cropped_plate: np.ndarray) -> np.ndarray:
    """
    Applies sequential image processing filters to enhance text visibility.
    
    Args:
        cropped_plate (np.ndarray): The raw BGR image crop from the YOLO detection.
        
    Returns:
        np.ndarray: A preprocessed, single-channel (grayscale/binary) image ready for OCR.
    """
    # I. Convert BGR (OpenCV default) to Grayscale
    gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
    #  A. Convert to single channel (grayscale) for OCR
    #  Y =  0.2126 * R + 0.7152 * G + 0.0722 * B
    
    # II. Apply CLAHE to enhance local contrast, especially useful for shadows or glare
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray) #  Divde by 255 to normalize to [0, 1] range for better contrast enhancement
    
    # III. Apply a slight Gaussian Blur to remove high-frequency noise before binarization
    blurred = cv2.GaussianBlur(enhanced_gray, (3, 3), 0)
    #  G(x,y) = (1/(2*pi*sigma^2)) * e^(-(x^2 + y^2)/(2*sigma^2))
    #  where sigma is the standard deviation of the Gaussian kernel, and (x, y) are the coordinates of the kernel relative to its center.
    
    # IV. Adaptive Thresholding to binarize the image (black and white)
    # The block size is 11, and the constant C subtracted from the mean is 2.
    binary = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 
        2
    )
    # T(x,y) =  Gaussian(x,y) * C + mean
    # if  p(x,y) > T(x,y) then pixel(x,y) = 1 else pixel(x,y) = 0
    
    return binary


def extract_plate_crop(frame: np.ndarray, bbox: list[int], padding: float = 0.12) -> np.ndarray:
    """
    Safely extracts the bounding box region from the main video frame.
    
    Args:
        frame (np.ndarray): The full video frame.
        bbox (list[int]): Bounding box coordinates in format [x_min, y_min, x_max, y_max].
        padding (float): Relative padding to add around the crop (default is 12%).
        
    Returns:
        np.ndarray: The cropped image tensor.
    """
    x_min, y_min, x_max, y_max = [int(coord) for coord in bbox]
    
    # [BUG FIX] Extract spatial dimensions (Height, Width) from the frame tensor
    height, width = frame.shape[:2]
    
    # Calculate dynamic padding based on the bounding box dimensions
    pad_x = int((x_max - x_min) * padding)
    pad_y = int((y_max - y_min) * padding)
    
    # Apply padding while ensuring coordinates do not fall out of the frame boundaries
    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(width, x_max + pad_x)
    y_max = min(height, y_max + pad_y)
    
    # Slice the numpy array with the expanded coordinates
    return frame[y_min:y_max, x_min:x_max]
