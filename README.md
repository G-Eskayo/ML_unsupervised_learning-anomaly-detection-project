# ML Anomaly Detection in IoT Network Traffic

This project applies **unsupervised machine learning** to the **CTU-IoT Malware dataset**, with the goal of identifying anomalous network connections that may indicate **malware activity**. It demonstrates a full workflow of **EDA, preprocessing, feature engineering, and anomaly detection** for cybersecurity applications.

---

## 🎯 Project Overview

- **Goal:** Detect malicious traffic in IoT network logs using unsupervised ML.
- **Dataset:** [CTU-IoT Malware Dataset (Kaggle)](https://www.kaggle.com/datasets/agungpambudi/network-malware-detection-connection-analysis)
- **Motivation:** Network logs contain millions of connections, but only a small fraction may be malicious. Detecting anomalies in high-dimensional data is a core cybersecurity challenge.

---

## 🛠️ Features

- **Exploratory Data Analysis (EDA):**
    - Inspected 23 features across ~23,000 network entries.
    - Analyzed categorical (IPs, ports, labels) and numerical attributes (packet size, duration).
- **Data Preprocessing:**
    - Dropped irrelevant/missing features.
    - Encoded categorical variables (IPs, ports).
    - Standardized numeric features with `StandardScaler`.
- **Feature Engineering:**
    - Created rolling counts of connections by source IP within 10-second windows.
    - Converted protocol-level details into categorical indicators.
- **Unsupervised Learning:**
    - Applied clustering and anomaly detection methods to group traffic and flag suspicious outliers.
- **Pipeline Design:**
    - Built scikit-learn pipelines with `ColumnTransformer` to unify preprocessing and modeling.

---

## 🤖 Tech Stack

- **Language:** Python 3.x
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn
- **Tools:** Jupyter Notebook

---

## 🚀 How to Run

git clone https://github.com/yourusername/iot-anomaly-detection.git
cd iot-anomaly-detection
pip install -r requirements.txt
jupyter notebook anomaly_detection.ipynb


---

## 📊 Example Outputs

- Correlation heatmaps for numeric features
- Connection distribution by IP/port
- Cluster assignments for malicious vs benign traffic
- Anomaly scores highlighting unusual connections

## ✅ Skills Demonstrated

- Machine Learning (clustering, anomaly detection)
- Data preprocessing & feature engineering for network traffic
- Cybersecurity analytics (malicious traffic detection)
- Model pipeline construction (scikit-learn)
- Visualization & interpretation of anomalies

---

## 📂 Project Structure

iot-anomaly-detection/

├── anomaly_detection.ipynb # main notebook

├── requirements.txt # dependencies

├── report.pdf # optional project report

└── README.md # documentation

