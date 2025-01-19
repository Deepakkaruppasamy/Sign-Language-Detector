import cv2
from ultralytics import YOLO


model = YOLO(r"C:\ML project\best (1).pt")  

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        for box in result.boxes:
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            confidence = box.conf[0]
            class_id = int(box.cls[0])

        
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{model.names[class_id]} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    
    cv2.imshow("Sign Language Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        print("Escape hit, closing...")
        break

cap.release()
cv2.destroyAllWindows()