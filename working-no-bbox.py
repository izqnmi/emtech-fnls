from roboflow import Roboflow
import cv2

api_key = "Dd3DTN4ns36a6fyImyK9"
rf = Roboflow(api_key=api_key)

workspace_name = "zzz-x9kiy"
project_name = "ambulance-detection-m9kn3"
workspace = rf.workspace(workspace_name)
project = workspace.project(project_name)
latest_version = max(project.versions(), key=lambda v: v.id)

if latest_version is None:
    print("Error: No versions found for the project.")
else:
    model = latest_version.model
    if model is None:
        print("Error: No model found for the latest version.")
    else:
        video_path = "C://Users//Kurt//Downloads//ambulance2.webm"  # replace with proper dir pls
        cap = cv2.VideoCapture(video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            response = model.predict(frame, confidence=40, overlap=30)

            if response is not None:
                print(response.json())
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                print("Error: Failed to get a response from the model.")
                
        cap.release()
        cv2.destroyAllWindows()
