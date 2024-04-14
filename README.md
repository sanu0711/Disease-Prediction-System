# Disease Prediction System

![GitHub last commit](https://img.shields.io/github/last-commit/your_username/your_repository_name)
![GitHub repo size](https://img.shields.io/github/repo-size/your_username/your_repository_name)

## Overview

This project implements a Disease Prediction System using various machine learning algorithms to predict potential diseases based on user-provided symptoms. The system utilizes a Django web framework to provide a user-friendly interface for inputting symptoms and viewing the predicted disease.

## Features

- **User Input**: Allows users to input their symptoms.
- **Prediction**: Utilizes machine learning algorithms including SVM, KNN, Naive Bayes, Random Forest, and Decision Tree to predict potential diseases based on the input symptoms.
- **Web Interface**: Implemented using Django, providing a seamless user experience.

## Installation

1. Clone the repository:

    ```
    git clone https://github.com/your_username/your_repository_name.git
    ```

2. Navigate to the project directory:

    ```
    cd your_repository_name
    ```

3. Install dependencies:

    ```
    pip install -r requirements.txt
    ```

    The following libraries are used in this project:
    - joblib
    - scikit-learn (includes DecisionTreeClassifier, SVC, RandomForestClassifier, GaussianNB, and KNeighborsClassifier)
    - pandas

4. Run the Django server:

    ```
    python manage.py runserver
    ```

5. Access the application at `http://localhost:8000` in your web browser.

## Usage

1. Access the web interface in your browser.
2. Input the symptoms you're experiencing.
3. Click on the "Predict" button to get the predicted disease.
4. View the predicted disease along with its probability.

## Screenshots

![Screenshot 1](screenshots/screenshot1.png)
*Description of the screenshot*

![Screenshot 2](screenshots/screenshot2.png)
*Description of the screenshot*

## Algorithms Used

- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Random Forest
- Decision Tree

## Contributors

- [Your Name](https://github.com/your_username)


