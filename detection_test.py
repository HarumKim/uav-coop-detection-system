import cv2
import os
import glob
from ultralytics import YOLO

def play_frames_with_yolo(frames_path, model_path, fps=30):
    # Load YOLO model
    print("Loading model:", model_path)
    model = YOLO(model_path)

    # Get list of image files
    frames = sorted(
        glob.glob(os.path.join(frames_path, "*.jpg")) +
        glob.glob(os.path.join(frames_path, "*.png"))
    )

    if not frames:
        print("No frames found in the folder.")
        return

    print(f"📸 Loaded {len(frames)} frames. Running YOLO detection...")

    delay = int(1000 / fps)  # ms per frame

    for frame_file in frames:
        frame = cv2.imread(frame_file)

        if frame is None:
            print(f"Could not read: {frame_file}")
            continue

        # Run YOLO inference
        results = model(frame, verbose=False)

        # YOLO result → get a plotted image with boxes
        annotated_frame = results[0].plot()

        # Show frame
        cv2.imshow("YOLO Detection", annotated_frame)

        # Quit on Q
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            print("Stopped by user.")
            break

    cv2.destroyAllWindows()


# --------------------------------------------------
# Run it
# --------------------------------------------------
if __name__ == "__main__":
    frames_path = r"C:\Users\Kim\Tec\Documents\Carrera_IRS\7th Semester\FinalChallenge\framesFUEGO3"

    model_path = r"C:\Users\Kim\Tec\Documents\Carrera_IRS\7th Semester\FinalChallenge\best.pt"
    play_frames_with_yolo(frames_path, model_path, fps=30)
