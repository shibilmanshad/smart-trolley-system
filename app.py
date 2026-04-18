from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import pandas as pd
import time
import atexit
import os
from pyzbar.pyzbar import decode
from ultralytics import YOLO

app = Flask(__name__)

# =============================
# LOAD YOLO MODEL (AI MODULE)
# =============================
model_path = "runs/detect/smart_trolley_model5/weights/best.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"YOLO model not found at: {model_path}")

yolo_model = YOLO(model_path)

# =============================
# LOAD PRODUCT CSV (BILLING)
# FIX 1: force product_id as string
# so "P1109" matches correctly
# =============================
products_df = pd.read_csv("ml_products.csv", dtype={'product_id': str})

# =============================
# BILL & BUDGET DATA
# =============================
bill = {}        # { product_id: qty }
total_amount = 0
budget_limit = 0

# =============================
# CAMERA SETUP
# =============================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# FIX 3: release camera cleanly on app exit
@atexit.register
def release_camera():
    cap.release()

if not cap.isOpened():
    raise RuntimeError("Could not open camera. Check if webcam is connected.")

last_scanned = {}
SCAN_COOLDOWN = 3  # seconds


# =============================
# GET PRODUCT BY PRODUCT_ID
# FIX 2: strip whitespace & force
# string so barcode matches CSV
# =============================
def get_product_by_id(product_id):
    product_id = str(product_id).strip()   # ← key fix
    row = products_df[products_df['product_id'] == product_id]

    if not row.empty:
        return (
            row.iloc[0]['name'],
            row.iloc[0]['category'],
            int(row.iloc[0]['price'])
        )

    return "Unknown", "Unknown", 0


# =============================
# VIDEO STREAM + YOLO + BARCODE
# =============================
def generate_frames():
    global total_amount

    while True:
        success, frame = cap.read()
        if not success:
            break

        current_time = time.time()

        # --------- YOLO (AI Detection) ---------
        yolo_results = yolo_model(frame, stream=True)
        for r in yolo_results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                class_name = yolo_model.names[cls_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"AI: {class_name}",
                            (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # --------- BARCODE (BILLING) ---------
        # Multi-pass decoding: try several preprocessed versions of the frame
        # This is essential when scanning barcodes shown on a phone screen
        # through a laptop camera (glare, angle, low contrast)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Pass 1: plain grayscale
        barcodes = decode(gray)

        # Pass 2: sharpened — helps with blurry phone screens
        if not barcodes:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened = cv2.filter2D(gray, -1, kernel)
            barcodes = decode(sharpened)

        # Pass 3: adaptive threshold — handles glare and uneven lighting
        if not barcodes:
            thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 51, 11
            )
            barcodes = decode(thresh)

        # Pass 4: upscaled — helps when barcode is small in frame
        if not barcodes:
            upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            barcodes = decode(upscaled)

        for barcode in barcodes:
            barcode_data = barcode.data.decode("utf-8").strip()

            # Cooldown to avoid duplicate scans
            if barcode_data in last_scanned:
                if current_time - last_scanned[barcode_data] < SCAN_COOLDOWN:
                    continue

            last_scanned[barcode_data] = current_time

            name, category, price = get_product_by_id(barcode_data)

            # Update bill
            bill[barcode_data] = bill.get(barcode_data, 0) + 1
            total_amount = sum(
                get_product_by_id(pid)[2] * qty
                for pid, qty in bill.items()
            )

            # Draw barcode box
            x, y, w, h = barcode.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"SCAN: {barcode_data} | {name} Rs.{price}"
            cv2.putText(frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# =============================
# ROUTES
# =============================
@app.route('/', methods=['GET', 'POST'])
def index():
    global budget_limit

    if request.method == 'POST':
        try:
            budget_limit = int(request.form.get('budget'))
        except:
            budget_limit = 0

    bill_items = []
    for pid, qty in bill.items():
        name, category, price = get_product_by_id(pid)
        subtotal = price * qty
        bill_items.append({
            "product_id": pid,
            "name": name,
            "category": category,
            "qty": qty,
            "price": price,
            "subtotal": subtotal
        })

    return render_template(
        'index.html',
        bill_items=bill_items,
        total=total_amount,
        budget=budget_limit
    )


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/bill_data')
def bill_data():
    bill_items = []
    for pid, qty in bill.items():
        name, category, price = get_product_by_id(pid)
        subtotal = price * qty
        bill_items.append({
            "product_id": pid,
            "name": name,
            "category": category,
            "qty": qty,
            "price": price,
            "subtotal": subtotal
        })

    return jsonify({
        "items": bill_items,
        "total": total_amount,
        "budget": budget_limit
    })


@app.route('/reset')
def reset_bill():
    global bill, total_amount
    bill = {}
    total_amount = 0
    return jsonify({"status": "reset"})


# =============================
# START APP
# FIX 4: debug=False prevents Flask
# from opening the camera twice
# =============================
if __name__ == "__main__":
    app.run(debug=False)
