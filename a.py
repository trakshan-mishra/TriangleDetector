import cv2 as cv
import numpy as np

def real_time_shape(show):
    # VIDEO CAPTURE
    cap_video = cv.VideoCapture(0)

    # RUNS FOREVER
    while True:
        ret, frame = cap_video.read()
        if not ret:
            break

        # Apply shape detection
        shapes = shapeDetector(frame)
        if show:
            # Display original
            cv.imshow('Original Image', frame)

            # Display shape output
            cv.imshow('Shapes', shapes)
        if cv.waitKey(5) & 0xFF == ord('q'):
            break

    cap_video.release()
    cv.destroyAllWindows()

def shapeDetector(image):
    # Convert to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edged = cv.Canny(blurred, 30, 150)  # Adjusted thresholds

    # Debug: show edged image
    cv.imshow('Edged', edged)

    # Find contours in the edge map
    contours, _ = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Debug: print number of contours found
    print(f'Number of contours found: {len(contours)}')

    for cnt in contours:
        shape = detect(cnt)
        if shape == "triangle":
            M = cv.moments(cnt)
            if M["m00"] != 0:
                cX = int((M["m10"] / M["m00"]))
                cY = int((M["m01"] / M["m00"]))
            else:
                cX, cY = 0, 0
            cv.drawContours(image, [cnt], -1, (34, 0, 156), 2)
            cv.putText(image, shape, (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return image

def detect(c):
    shape = "unidentified"
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.04 * peri, True)
    if len(approx) == 3:
        shape = "triangle"
    return shape

if __name__ == "__main__":
    real_time_shape(1)

