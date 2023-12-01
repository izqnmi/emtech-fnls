from roboflow import Roboflow
import cv2

# Set your Roboflow API key
api_key = "Dd3DTN4ns36a6fyImyK9"
rf = Roboflow(api_key=api_key)

# Set your workspace and project names
workspace_name = "zzz-x9kiy"
project_name = "ambulance-detection-m9kn3"

# Get the workspace and project
workspace = rf.workspace(workspace_name)
project = workspace.project(project_name)

# Get the latest version in the project
latest_version = max(project.versions(), key=lambda v: v.id)

# Check if the latest version exists
if latest_version is None:
    print("Error: No versions found for the project.")
else:
    # Get the model from the latest version
    model = latest_version.model

    # Check if the model exists
    if model is None:
        print("Error: No model found for the latest version.")
    else:
        # Open a video file
        video_path = "C://Users//Kurt//Downloads//ambulance2.webm"  # Replace with the path to your video file
        cap = cv2.VideoCapture(video_path)

        while True:
            # Read a frame from the video
            ret, frame = cap.read()

            # Break the loop if the video has ended
            if not ret:
                break

            # Infer on the current frame
            response = model.predict(frame, confidence=40, overlap=30)

            # Check if response is successful
            if response is not None:
                # Process the response as needed
                print(response.json())

                # Display the current frame (optional, for visualization purposes)
                cv2.imshow("Frame", frame)

                # Break the loop if the 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                print("Error: Failed to get a response from the model.")

        # Release the video capture object and close any open windows
        cap.release()
        cv2.destroyAllWindows()
