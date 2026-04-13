# 📊 REX-Cluster-V1 | Customer Segmentation Monolith

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-blue?logo=plotly)
![License](https://img.shields.io/badge/License-MIT-green)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open%20App-brightgreen?logo=rocket)](https://kmeansclusteringbychinmay.streamlit.app/)

</p>

---

## 🚀 Overview

**REX-Cluster-V1** is an advanced **customer segmentation dashboard** built with Streamlit that leverages **K-Means clustering** to identify distinct customer groups based on spending behavior.

Designed as a **vertical AI analytics engine**, it provides interactive insights into customer patterns using modern UI/UX and real-time clustering adjustments.

---

## ✨ Features

* 🤖 K-Means clustering (dynamic K selection)
* 🎛️ Interactive control panel (cluster tuning)
* 📊 Real-time segmentation visualization
* ⭐ Centroid highlighting (cluster centers)
* 📈 Elbow method for optimal K selection
* 📋 Cluster distribution insights
* 🔍 Raw data preview
* 🎨 Premium UI (dark theme + neon + glassmorphism)

---

## 🛠️ Tech Stack

* Python 3.x
* Streamlit
* Pandas
* Plotly (Express + Graph Objects)
* Scikit-learn

---

## 📂 Project Structure

```bash id="rexstruct1"
.
├── app.py
├── Mall_Customers.csv
├── requirements.txt
├── README.md
```

---

## ⚙️ Installation

```bash id="rexinstall1"
git clone https://github.com/your-username/rex-cluster-dashboard.git
cd rex-cluster-dashboard
pip install -r requirements.txt
```

---

## ▶️ Run Locally

```bash id="rexrun1"
streamlit run app.py
```

---

## 📊 Model Details

* Algorithm: **K-Means Clustering**
* Initialization: k-means++
* Features:

  * Annual Income
  * Spending Score
* Output:

  * Customer Segments (Clusters)

---

## 📈 Visualizations

* 📊 Interactive scatter plot (clusters)
* ⭐ Cluster centroids (highlighted)
* 📉 Elbow curve (WCSS vs K)
* 📋 Cluster distribution table

---

## 🧠 How It Works

1. Dataset is loaded from CSV
2. Relevant features are selected
3. K-Means clustering is applied
4. Customers are grouped into segments
5. Results are visualized dynamically

---

## 📁 Dataset

The dataset (`Mall_Customers.csv`) includes:

* CustomerID
* Gender
* Age
* Annual Income (k$)
* Spending Score (1–100)

---

## 🎛️ Controls

* **K Value Slider** → Adjust number of clusters
* **Elbow Analysis Toggle** → Find optimal K

---

## 🚀 Deployment

Deploy easily using **Streamlit Cloud**:

1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Create new app
4. Deploy 🎉

---

## 🔮 Future Improvements

* 🧠 Auto cluster labeling (High Value / Low Value)
* 📊 Customer persona generation
* 📉 PCA visualization (2D/3D)
* 📊 Cluster comparison dashboard
* 🔍 Interactive drill-down

---

## 👨‍💻 Author

**Chinmay V Chatradamath**

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
