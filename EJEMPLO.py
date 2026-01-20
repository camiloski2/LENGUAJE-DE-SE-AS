import cv2
import mediapipe as mp

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Función para saber si un dedo está levantado
def fingers_up(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Pulgar
    if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Otros dedos
    for id in range(1, 5):
        if hand_landmarks.landmark[tips_ids[id]].y < hand_landmarks.landmark[tips_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

# Traducir a letra (simplificado)
def detect_letter(fingers):
    if fingers == [0, 0, 0, 0, 0]:
        return "A"
    elif fingers == [0, 1, 1, 1, 1]:
        return "B"
    elif fingers == [0, 1, 0, 0, 0]:
        return "L"
    elif fingers == [0, 1, 1, 0, 0]:
        return "F"
    elif fingers == [1, 1, 1, 1, 1]:
        return "5"
    else:
        return ""

# Captura
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            finger_states = fingers_up(hand_landmarks)
            letra = detect_letter(finger_states)

            if letra:
                cv2.putText(img, f"Letra: {letra}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (0, 255, 0), 3)

    cv2.imshow("Traductor de Señas (Demo)", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
