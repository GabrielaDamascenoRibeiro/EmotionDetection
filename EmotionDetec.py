import cv2
from fer import FER

detector = FER(mtcnn=True)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    result = detector.detect_emotions(frame)
    
    for face in result:
        (x, y, w, h) = face["box"]
        emotions = face["emotions"]
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        emotion = max(emotions, key=emotions.get)
        score = emotions[emotion]
        
        text = f"{emotion}: {score:.2f}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    cv2.imshow('Emotion Detection', frame)
    
    # Break the loop on 'q'
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
