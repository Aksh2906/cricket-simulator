import cv2
import mediapipe as mp
import socket

UDP_IP = "127.0.0.1"   # Unity running on same laptop
UDP_PORT = 5060

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    height_value = 0.5  # default mid height

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            h, w, c = img.shape

            # Use wrist landmark (id 0)
            wrist = handLms.landmark[0]
            height_value = 1 - wrist.y   # invert (top = 1, bottom = 0)

    message = str(height_value).encode()
    sock.sendto(message, (UDP_IP, UDP_PORT))

    cv2.imshow("MediaPipe", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
