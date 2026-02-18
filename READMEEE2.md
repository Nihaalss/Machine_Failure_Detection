# 🛠️ Machine Failure Prediction

This project focuses on predicting machine failure using sensor data. It includes data preprocessing, exploratory data analysis (EDA), and model building using a Random Forest classifier in Python.

## 📁 Dataset

The dataset contains 944 records and 10 features:
- `footfall`
- `tempMode`
- `AQ` (Air Quality)
- `USS` (Ultrasonic Sensor)
- `CS` (Current Sensor)
- `VOC` (Volatile Organic Compounds)
- `RP` (Rotational Power)
- `IP` (Input Power)
- `Temperature`
- `fail` (Target: 1 if the machine failed, 0 otherwise)

## 📊 Exploratory Data Analysis

Performed using `pandas`, `matplotlib`, and `seaborn`:
- Correlation heatmaps
- Value distributions per sensor
- Relationship of each feature to machine failure
- Comparative analysis of sensor readings for failed vs. non-failed instances

## 🧠 Model Training

- **Model Used:** Random Forest Classifier (`sklearn.ensemble`)
- **Train/Test Split:** 75% training / 25% testing
- **Accuracy:** ~89.4%

### Classification Report:
| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| 0     | 0.90      | 0.90   | 0.90     |
| 1     | 0.88      | 0.89   | 0.89     |

## 🔍 Features Used for Training
All columns except `fail`.

## 🧪 Example Predictions
You can provide new sensor values and get real-time failure predictions using the trained model. Example inputs are included in the notebook.

## 🧰 Technologies Used

- Python
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn

## 📁 Folder Structure

