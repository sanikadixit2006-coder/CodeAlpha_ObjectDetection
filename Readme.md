# 🎯 Object Detection & Tracking
### CodeAlpha Artificial Intelligence Internship — Task Submission

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

A **professional real-time Object Detection and Tracking** system built with **Python**, **OpenCV**, and **YOLOv8**. The application captures live webcam feed, detects objects using the Open Images V7 model, tracks them across frames, and displays polished visual overlays.

---

## ✨ Key Features

| Feature | Details |
|---|---|
| 🔍 Advanced Model | `yolov8s-oiv7.pt` with 600+ Open Images V7 classes |
| 🎯 Improved Detection | 1280x720 inference for better small object detection |
| 🧹 Noise Filtering | skip boxes smaller than 20x20 pixels |
| 🌈 Unique Class Colors | stable color per class using HSV hashing |
| 📊 Live Counter | top-right panel shows object count in current frame |
| 📈 FPS Graph | live FPS history bar in top-left |
| 🧷 Tracking Trails | fading motion trails for tracked objects |
| 📸 Screenshot | press `S` to save a flash screenshot |
| ⏸ Pause Mode | press `P` to pause/resume detection |
| 🎚 Confidence Control | `+` / `-` to adjust confidence threshold live |
| 🆘 Help Overlay | press `H` to toggle hotkey help |
| 🧾 Clean Code | modular functions and beginner-friendly structure |

---

## 🛠️ Hotkeys

| Key | Action |
|---|---|
| `Q` | Quit cleanly |
| `S` | Save screenshot to current folder |
| `P` | Pause / Resume detection |
| `+` | Increase confidence threshold |
| `-` | Decrease confidence threshold |
| `H` | Toggle help overlay |

---

## 🚀 Installation & Setup

### Prerequisites
- Python **3.8+**
- A working **webcam**
- Internet access to download the YOLO model if not already present

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python Main.py
```

---

## 🔧 What Changed

- Upgraded model to `yolov8s-oiv7.pt`
- Inference resolution set to 1280x720
- Live confidence threshold control with `+` / `-`
- Unique persistent class colors with white label text
- Object counter panel for visible classes
- Bottom statistics bar with model and object summary
- FPS history graph with real-time refresh
- Screenshot feature with white flash effect
- Pause mode with center red `PAUSED` text
- Tracking trail effect for each tracked ID
- Help overlay with hotkeys
- Clean function-based structure and beginner-friendly code

---

## 🗂️ Folder Structure

```
CodeAlpha_ObjectDetection/
│
├── Main.py            ← Main application entry point
├── Requirements.txt   ← Python dependency list
└── Readme.md          ← Project documentation
```

---

## ❗ Notes

- Keep `Requirements.txt` unchanged: only `ultralytics` and `opencv-python` are required.
- The app prints a startup banner and provides clean terminal feedback.
- The screenshot file format is `screenshot_YYYYMMDD_HHMMSS.jpg`.

---

## 📄 License

This project is submitted for the **CodeAlpha AI Internship**. Feel free to explore and learn from the implementation.

---

*Built with care for a professional object detection and tracking demo.*
