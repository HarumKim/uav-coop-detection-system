# 🚁 Cooperative UAV Fire Detection System  
**7th Semester | Autonomous Systems & Artificial Intelligence**

This repository features a **cooperative multi-UAV system** designed for **autonomous fire detection, mapping, and precision inspection**. By leveraging **computer vision** and **coordinated navigation**, the system automates the workflow of identifying fire hazards across large areas with style and efficiency.

---

## 📌 Project Overview

The system uses a **dual-drone architecture**, where UAVs operate sequentially to bridge the gap between **wide-area surveillance** and **close-range inspection**:

- **Drone 1 | The Scout**  
  Executes a structured grid-search pattern using a custom **YOLO model** to detect fire and log precise coordinates.

- **Drone 2 | The Inspector**  
  Receives the logged data and performs **targeted navigation**, **visual servoing**, and **high-altitude inspection** of identified hotspots.

---

## ✨ Key Features

- 🔥 **Real-Time Fire Detection**  
  Integrated **Ultralytics YOLO** model optimized for fire signature recognition with real-time bounding box visualization.

- 🗺️ **Autonomous Grid Mapping**  
  Drone 1 performs systematic exploration using logical cell coordinates and multi-frame validation to reduce false positives.

- 📍 **Coordinate Logging System**  
  Automated data pipeline that stores validated fire locations in a shared registry for seamless mission handoff.

- 🧭 **Precision Geometric Navigation**  
  Drone 2 uses yaw alignment and Euclidean distance calculations for efficient path planning.

- 🎯 **Visual Servoing**  
  Advanced alignment logic using YOLO detections to dynamically center the UAV over the fire source.

- 🛬 **Automated Inspection Maneuvers**  
  Pre-programmed flight sequences for safe descent, stable hovering at inspection height, and ascent.

---

## 🛠️ Tech Stack

| Category | Tools |
|--------|------|
| **Languages** | Python |
| **AI / Computer Vision** | Ultralytics YOLO, OpenCV, NumPy |
| **Hardware / SDK** | DJI RoboMaster SDK (Tello Talent) |
| **Navigation** | Geometric Path Planning, Visual Servoing |

---

## ▶️ Getting Started

### 1. Environment Setup

This project uses **Conda** for environment management to ensure dependency stability.

```bash
# Create the environment from the YAML file
conda env create -f tello_drone.yaml

# Activate the environment
conda activate tello_drone
```

### 2. Drone Connection Protocol

The system supports **two connection modes**, depending on whether you want to run the **full cooperative mission** or test each UAV independently.

#### 🔄 Automatic Connection (Recommended)

Run the following script:

```bash
python connection_drones.py
```

## 🎥 Final Demonstration

The following videos showcase the **cooperative UAV fire detection system** in action, highlighting the sequential workflow between both drones.

- **Drone 1 | Scout – Fire Detection & Mapping**  
  Autonomous grid navigation, real-time fire detection, and coordinate logging.  
  🔗 [Watch Demo](https://tecmx-my.sharepoint.com/:v:/g/personal/a00836962_tec_mx/IQBRRjdpmKHFS5h7zM3pfhooAWF2SZ7QZbJd2dE3OB-iSyU?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=Wnxxlv)

- **Drone 2 | Inspector – Targeted Inspection**  
  Precision navigation, visual servoing, and high-altitude fire inspection using logged coordinates.  
  🔗 [Watch Demo](https://tecmx-my.sharepoint.com/:v:/g/personal/a00836962_tec_mx/IQDF9RkXAnMkTIokEMubxzl5ATMxyVx_aZqyFX2n-Rad2Gs?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=B3eWa3)
