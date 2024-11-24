import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# โหลดโมเดลและ Config สำหรับการตรวจจับใบหน้า
net = cv2.dnn.readNetFromCaffe('./caffemodel/deploy.prototxt', './caffemodel/res10_300x300_ssd_iter_140000.caffemodel')

# ฟังก์ชันสำหรับวาดข้อความภาษาไทย
def draw_thai_text(image, text, position, font_path="./fonts/angsana.ttc", font_size=32, color=(0, 0, 255)):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

# โหลด Face Recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('./IMG_FRIENDLY/face_model.yml')  # โหลดโมเดลการจดจำใบหน้า

# เปิดกล้อง (หรือใส่วิดีโอไฟล์)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # เตรียมภาพสำหรับ DNN
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)

    # ตรวจจับใบหน้า
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # กำหนด Threshold
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")

            # คำนวณจุดกึ่งกลางและรัศมีของวงกลม
            centerX = startX + (endX - startX) // 2
            centerY = startY + (endY - startY) // 2
            radius = max((endX - startX) // 2, (endY - startY) // 2)

            # วาดวงกลมรอบศีรษะ
            cv2.circle(frame, (centerX, centerY), radius, (0, 255, 0), 2)

            # ตรวจสอบใบหน้ากับฐานข้อมูล
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_region = gray[startY:endY, startX:endX]
            label, confidence = face_recognizer.predict(face_region)

            # ถ้ามีการจับคู่ใบหน้าในฐานข้อมูล
            if confidence < 70:  # เปลี่ยนค่าตามความเหมาะสม
                text = "Friendly"
                text_color = (0, 255, 0)  # สีเขียวสำหรับเพื่อน
            else:
                text = "Enemy"
                text_color = (255, 0, 0)  # สีแดงสำหรับศัตรู

            # วาดข้อความ
            text_position = (startX, startY - 30)
            frame = draw_thai_text(frame, text, text_position, font_size=50, color=text_color)

            # วาด crosshair (กากบาท)
            cv2.line(frame, (centerX - 1000, centerY), (centerX + 1000, centerY), (0, 0, 255), 2)
            cv2.line(frame, (centerX, centerY - 1000), (centerX, centerY + 1000), (0, 0, 255), 2)

    # แสดงภาพ
    cv2.imshow("Face Recognition with Thai Text", frame)

    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
