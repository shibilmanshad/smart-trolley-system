from flask import Flask, render_template, Response, jsonify, request
import cv2
import pandas as pd
import time
from pyzbar.pyzbar import decode
from ultralytics import YOLO

app = Flask(__name__)

# =============================
# LOAD YOLO MODEL (AI MODULE)
# =============================
yolo_model = YOLO("runs/detect/smart_trolley_model5/weights/best.pt")

# =============================
# LOAD PRODUCT CSV (BILLING)
# =============================
products_df = pd.read_csv("ml_products.csv")

# =============================
# BILL & BUDGET DATA
# =============================
bill = {}   # {product_id: qty}
total_amount = 0
budget_limit = 0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

last_scanned = {}
SCAN_COOLDOWN = 3  # seconds


# =============================
# GET PRODUCT BY PRODUCT_ID
# =============================
def get_product_by_id(product_id):
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
        barcodes = decode(frame)

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
            label = f"SCAN: {barcode_data} | {name} ₹{price}"
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
# =============================
if __name__ == "__main__":
    app.run(debug=True)
