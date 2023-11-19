import cv2
import pytesseract
import numpy as np

# Set Tesseract path (update with your Tesseract installation path)
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract'

# Load pre-trained Haar Cascade for license plate detection
cascade_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
cascade = cv2.CascadeClassifier(cascade_path)

# State codes dictionary
states = {
    "AN": "Andaman and Nicobar Islands",
  "AP": "Andhra Pradesh",
  "AR": "Arunachal Pradesh",
  "AS": "Assam",
  "BR": "Bihar",
  "CH": "Chandigarh",
  "CG": "Chhattisgarh",
  "DD": "Dadra and Nagar Haveli and Daman and Diu",
  "DL": "Delhi",
  "GA": "Goa",
  "GJ": "Gujarat",
  "HR": "Haryana",
  "HP": "Himachal Pradesh",
  "JK": "Jammu and Kashmir",
  "JH": "Jharkhand",
  "KA": "Karnataka",
  "KL": "Kerala",
  "LA": "Ladakh",
  "LD": "Lakshadweep",
  "MP": "Madhya Pradesh",
  "MH": "Maharashtra",
  "MN": "Manipur",
  "ML": "Meghalaya",
  "MZ": "Mizoram",
  "NL": "Nagaland",
  "OD": "Odisha",
  "PY": "Puducherry",
  "PB": "Punjab",
  "RJ": "Rajasthan",
  "SK": "Sikkim",
  "TN": "Tamil Nadu",
  "TS": "Telangana",
  "TR": "Tripura",
  "UP": "Uttar Pradesh",
  "UK": "Uttarakhand",
  "WB": "West Bengal"
}

def extract_license_plate(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform license plate detection
    plates = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in plates:
        plate = image[y:y+h, x:x+w]

        # Use Otsu's method for thresholding
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        _, plate_threshold = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Perform OCR on the license plate
        plate_text = pytesseract.image_to_string(plate_threshold, config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

        # Check if "IND" is present in the OCR result
        if "IND" in plate_text:
            # Extract state code
            state_code = plate_text[plate_text.find("IND")+3:plate_text.find("IND")+5]

            # Determine the state from the state codes dictionary
            state = states.get(state_code, 'Unknown State')

            return plate_text, state

    return None, None

# Open the webcam (change 0 to the camera index if you have multiple cameras)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Extract license plate information
    extracted_text, recognized_state = extract_license_plate(frame)

    # Display the result
    if extracted_text is not None:
        cv2.putText(frame, f"License Plate: {extracted_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"State: {recognized_state}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Real-Time License Plate Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()