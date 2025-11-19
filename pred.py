import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import numpy as np
from datetime import datetime
import csv
import os
from ultralytics import YOLO
import os

# -----------------------------
# KERAS MODELL LISTA
# -----------------------------
model_paths = [
    r"E:\PythonProjects\ocr\modellek\best_model800.h5",
    r"E:\PythonProjects\ocr\modellek\best_model1200.h5",
    r"E:\PythonProjects\ocr\modellek\best_model1600.h5",
    r"E:\PythonProjects\ocr\modellek\best_model2000.h5",
    r"E:\PythonProjects\ocr\modellek\best_model2400.h5",
    r"E:\PythonProjects\ocr\modellek\best_model10000.h5",
    r"E:\PythonProjects\ocr\modellek\best_modelSMALL_balanced_dataset_2400.h5",
]

keras_models = [(os.path.basename(p), tf.keras.models.load_model(p)) for p in model_paths]

# -----------------------------
# YOLO MODELL LISTA
# -----------------------------
yolo_paths = {
    "runs_yolo2400": r"E:\PythonProjects\ocr\modellek\runs_yolo2400\classify\train\weights\best.pt",
    "runs_yolo_mnist": r"E:\PythonProjects\ocr\modellek\runs_yolo_mnist\classify\train\weights\best.pt",
}

yolo_models = {name: YOLO(path) for name, path in yolo_paths.items()}

# -----------------------------
roi_area = (135, 145, 95, 40)
csv_path = "eredmenyek.csv"

def preprocess_frame(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 4
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 2))
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    eroded = cv2.erode(processed, kernel, iterations=3)
    return eroded

def extract_keras_digit(img, model):
    img_size = 28
    img_resized = cv2.resize(img, (img_size, img_size))
    img_np = img_to_array(img_resized).astype('uint8')
    img_array = img_np.reshape(1, img_size, img_size, 1).astype('float32') / 255.0
    prediction = model.predict(img_array, verbose=0)[0]
    digit = np.argmax(prediction)
    confidence = float(prediction[digit] * 100)
    return digit, confidence

def extract_yolo_digit(img, model, yolo_name):
    if (yolo_name == 'runs_yolo_mnist'):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    results = model(img, verbose=False)
    pred = results[0].probs

    digit = int(pred.top1)
    confidence = float(pred.top1conf * 100)
    return digit, confidence

# -----------------------------

cap = cv2.VideoCapture(1)
last_results = {}  # model_name → (predictions, confidences)

print("SPACE → felismerés futtatása (minden modellel)")
print("ENTER → mentés CSV-be")
print("Q → kilépés")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Nem sikerült képet beolvasni.")
        break

    padding = 5
    x, y, w, h = roi_area
    roi = frame[y:y + h, x:x + w]
    roi_for_model = roi.copy()
    processed = preprocess_frame(roi)

    # Kontúrok keresése
    contours, _ = cv2.findContours(processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_rois = []
    for c in contours:
        cx, cy, cw, ch = cv2.boundingRect(c)
        if cw > 5 and ch > 15:
            # Padding hozzáadása, figyelve a kép határait
            x1 = max(cx - padding, 0)
            y1 = max(cy - padding, 0)
            x2 = min(cx + cw + padding, roi.shape[1])
            y2 = min(cy + ch + padding, roi.shape[0])

            digit_rois.append((x1, y1, x2 - x1, y2 - y1))

    digit_rois = sorted(digit_rois, key=lambda b: b[0])

    # Bounding boxok kirajzolása a frame-re
    for (cx, cy, cw, ch) in digit_rois:
        cv2.rectangle(frame, (x + cx, y + cy), (x + cx + cw, y + cy + ch), (0, 255, 0), 2)

    # ROI keret rajzolása
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Ablakok megjelenítése
    cv2.imshow("Eredeti + ROI", frame)
    cv2.imshow("Kontúrozott ROI", processed)

    key = cv2.waitKey(1) & 0xFF

    # -----------------------------
    # SPACE → futtat minden modellen
    # -----------------------------
    if key == ord(' '):
        user_number = input("Melyik szám van a képen: ")
        last_results.clear()

        keras_model_images = []
        # Keras modellek
        for model_name, model in keras_models:
            predictions = []
            confidences = []

            for idx, (cx, cy, cw, ch) in enumerate(digit_rois):
                digit_img = roi_for_model[cy:cy+ch, cx:cx+cw]
                gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)

                keras_model_images.append((idx, gray))

                digit, conf = extract_keras_digit(gray, model)
                predictions.append(digit)
                confidences.append(conf)

            last_results[model_name] = (predictions, confidences)

            print(f"\n[Keras] {model_name}")
            print("  Eredmény:", ''.join(map(str, predictions)))
            print("  Pontosság:", [f"{c:.2f}%" for c in confidences])

            '''if model_name == "best_model10000.h5":
                for idx, img in keras_model_images:
                    try:
                        show_img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_NEAREST)
                        win_name = f"KERAS input {idx + 1}"
                        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                        cv2.imshow(win_name, show_img)
                    except Exception as e:
                        print("Hiba keras kép megjelenítésénél:", e)
            '''


        # YOLO modellek
        for yolo_name, model in yolo_models.items():
            predictions = []
            confidences = []

            # ha runs_yolo2400 → képek megjelenítése szükséges
            yolo_input_images = []  # ide mentjük a képeket csak annál a modellnél

            for idx, (cx, cy, cw, ch) in enumerate(digit_rois):
                digit_img = roi_for_model[cy:cy+ch, cx:cx+cw]
                # runs_yolo_mnist → szürke kép a YOLO-nak
                if yolo_name == 'runs_yolo_mnist':
                    digit_img = preprocess_frame(digit_img)

                # runs_yolo2400 → EREDETI BGR megy, de eltároljuk és megjelenítjük
                if yolo_name == 'runs_yolo2400':
                    # mentés a listába
                    yolo_input_images.append((idx, digit_img))

                digit, conf = extract_yolo_digit(digit_img, model, yolo_name)
                predictions.append(digit)
                confidences.append(conf)

            last_results[yolo_name] = (predictions, confidences)

            print(f"\n[YOLO] {yolo_name}")
            print("  Eredmény:", ''.join(map(str, predictions)))
            print("  Pontosság:", [f"{c:.2f}%" for c in confidences])

            # ---------------------------------------------------------
            #  >>> Megjelenítés runs_yolo2400 modell esetén <<<
            # ---------------------------------------------------------
            '''if yolo_name == "runs_yolo2400":
                for idx, img in yolo_input_images:
                    try:
                        show_img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_NEAREST)
                        win_name = f"YOLO2400 input {idx + 1}"
                        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                        cv2.imshow(win_name, show_img)
                    except Exception as e:
                        print("Hiba YOLO2400 kép megjelenítésénél:", e)
            '''

    # -----------------------------
    # ENTER → CSV mentés
    # -----------------------------
    if key == 13:
        if not last_results:
            print("Nincs elvégzett felismerés.")
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            roi_base_path = os.path.join("eredmenyek", timestamp)
            os.makedirs(roi_base_path, exist_ok=True)

            # Mentjük az eredeti ROI-t
            roi_path = os.path.join(roi_base_path, "roi.png")
            cv2.imwrite(roi_path, roi_for_model)

            # Mentés CSV-be
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=';')

                if not file_exists:
                    writer.writerow(["timestamp", "model", "prediction", "user_number", "result", "confidences"])

                for model_name, (pred, confs) in last_results.items():
                    prediction_str = ''.join(map(str, pred))
                    conf_strs = [f"{c:.2f}" for c in confs]
                    result = "OK"
                    if (prediction_str != user_number):
                        result = 'NOK'
                    writer.writerow([timestamp, model_name, prediction_str, user_number, result] + conf_strs)

            # -----------------------------
            # Képek mentése
            # -----------------------------
            for model_name, (pred, confs) in last_results.items():
                model_folder = os.path.join(roi_base_path, model_name)
                os.makedirs(model_folder, exist_ok=True)

                if "keras" in model_name.lower():
                    # Keras képek
                    for idx, (cx, cy, cw, ch) in enumerate(digit_rois):
                        digit_img = roi_for_model[cy:cy + ch, cx:cx + cw]
                        gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                        save_path = os.path.join(model_folder, f"{model_name}_{idx + 1}.png")
                        cv2.imwrite(save_path, gray)

                else:
                    # YOLO képek
                    for idx, (cx, cy, cw, ch) in enumerate(digit_rois):
                        digit_img = roi_for_model[cy:cy + ch, cx:cx + cw]
                        if model_name == 'runs_yolo_mnist':
                            digit_img = preprocess_frame(digit_img)
                        save_path = os.path.join(model_folder, f"{model_name}_{idx + 1}.png")
                        cv2.imwrite(save_path, digit_img)

            print(f"\nCSV és képek mentve a {roi_base_path} mappába!")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
