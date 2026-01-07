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
