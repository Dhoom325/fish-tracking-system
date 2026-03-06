import cv2
import os

# Find a video file in the folder
video_file = None
for file in os.listdir("."):
    if file.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        video_file = file
        break

if video_file is None:
    print("No video file found in this folder. Please add a video.")
    exit()

print(f" Using video: {video_file}")
cap = cv2.VideoCapture(video_file)

# Background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

frame_count = 0

# Optional: save processed video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    frame_count += 1

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fish_count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:  # filter noise
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            fish_count += 1

    # Display fish count
    cv2.putText(frame, f"Total Fish: {fish_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Fish Detection", frame)

    # Save first frame dimensions
    if out is None:
        out = cv2.VideoWriter("output.mp4", fourcc, 20, (frame.shape[1], frame.shape[0]))
    out.write(frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
