# 🛒 Smart Trolley System (YOLO + Barcode)

A smart supermarket trolley system that uses **YOLOv8 object detection** and **barcode scanning (pyzbar)** to automatically detect products, calculate prices, and show a real-time bill preview before checkout.

## 🚀 Features
- Real-time product detection using YOLOv8
- Barcode scanning using pyzbar
- Automatic bill generation
- Beep sound on successful scan
- Budget limit input and warning
- Web-based dashboard using Flask
- CSV-based product database
- Real-time video stream

## 🧠 Technologies Used
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Flask
- pyzbar
- Pandas
- HTML/CSS/JavaScript

## 📁 Project Structure
smart-trolley-system/
├── app.py
├── ml_products.csv
├── requirements.txt
├── README.md
├── .gitignore
├── static/
│ └── beep.mp3
├── templates/
│ └── index.html
└── dataset/
└── data.yaml

## 📊 Dataset
Due to large file size, the YOLO training dataset (train/valid/test images) is not included in this repository.  
Only the `data.yaml` configuration file is provided.

## ⚙️ How to Run
```bash
pip install -r requirements.txt
python app.py

Open in browser:

http://127.0.0.1:5000

💡 Use Case

This project is designed for:

Smart retail systems

Computer vision based billing

AI-based supermarket automation

Academic mini/final year projects

👨‍💻 Author

Shibil Manshad