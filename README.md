# Smart Crop Suggestion System

A machine-learning-based web application that recommends suitable crops based on soil nutrients and environmental conditions.

## Features

- Crop recommendations using a trained Random Forest model.
- Input fields for nitrogen (N), phosphorus (P), potassium (K), temperature, humidity, soil pH, and rainfall.
- Web interface built with Flask and HTML templates.
- Model health-check API endpoint.
- Dataset exploration and model training scripts.

## Technologies Used

- Python
- Flask
- HTML
- CSS
- Pandas
- NumPy
- Scikit-learn
- Joblib

## Project Structure

```text
smart-crop-suggestion-system/
├── app.py
├── crop_model.pkl
├── Crop_recommendation.csv
├── requirements.txt
├── train_model.py
├── explore_dataset.py
├── test_prediction.py
├── templates/
│   ├── index.html
│   ├── predict.html
│   └── result.html
├── .env.example
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd smart-crop-suggestion-system
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

## Health Check

Visit this endpoint to check whether the application and ML model are working:

```text
http://127.0.0.1:5000/api/health
```

## Machine Learning Model

The project uses a Random Forest classifier trained on the crop recommendation dataset.

The reported test accuracy during development was 99.55%. This is a held-out dataset result and does not guarantee the same accuracy in real farming conditions.

## Future Improvements

- Top-three crop recommendations.
- Crop recommendation explanations.
- Weather API integration.
- Indian-language support.
- Farmer-friendly dashboard.

## Project Status

Under development.