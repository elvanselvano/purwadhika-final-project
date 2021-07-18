<p align="center">
  <a href="" rel="noopener">
 <img src="https://github.com/elvanselvano/purwadhika-final-project/blob/main/assets/wallpaper.jpg" alt="Project logo"></a>
</p>
<h3 align="center">Washington D.C. Residential Properties Price Prediction 🏠</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
![Code Size](https://img.shields.io/github/languages/code-size/elvanselvano/purwadhika-final-project)
![Contributors](https://img.shields.io/github/contributors/elvanselvano/purwadhika-final-project)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

</div>

---

<p align="center"> This web app uses actual transaction data and machine learning models to predict the prices of housing in Washington, D.C.. The transaction data contain attributes and transaction prices of residential property that respectively serve as independent variables and dependent variables for the machine learning models.
    <br> 
</p>

## 📝 Table of Contents

- [Problem Statement](#problem_statement)
- [Idea / Solution](#idea)
- [Setting up a local environment](#getting_started)
- [Technology Stack](#tech_stack)
- [Authors](#authors)
- [Acknowledgments](#acknowledgments)

## 🧐 Problem Statement <a name = "problem_statement"></a>

The housing market is one of the most crucial components of any national economy. Hence, observations of the housing market and accurate predictions of real estate prices are helpful for real estate buyers and sellers as well as economic specialists. It is crucial for the establishment of real estate policies and can help real estate owners and agents make informative decision. However, real estate forecasting is a complicated and difficult task owing to many direct and indirect factors that inevitably influence the accuracy of predictions.

## 💡 Idea / Solution <a name = "idea"></a>

The main idea is to train a regression model using historical transactions. The output of this model is the predicted price of a house given its features. It will also give the range of price by 95% predictions interval which means given a prediction of ‘y’ given ‘x’, there is a 95% likelihood that the range ‘a’ to ‘b’ covers the true outcome. In order to measure the performance of the model, we focus on Mean Absolute Error (MAE) because it is robust to outliers.

## 🏁 Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Make sure that you have [Python 3.6 - Python 3.8](https://www.python.org/downloads/release/python-386/) installed. The libraries required to run this project is in the `requirements.txt`. You can install them using [PIP](https://pip.pypa.io/en/stable/installing/).

```
pycaret==2.3.2
catboost>=0.23.2
pandas==1.1.5
streamlit
```

### Running
In order to run a [streamlit](https://streamlit.io/) app, all you need to do is write the following command.

```
streamlit run app.py
```

That’s it! In the next few seconds the app will open in a new tab in your default browser.

## ⛏️ Built With <a name = "tech_stack"></a>

![Deployment](https://github.com/elvanselvano/purwadhika-final-project/blob/main/assets/deployment.png)

## ✍️ Authors <a name = "authors"></a>

- [@mini4wd14](https://github.com/mini4wd14) - Problem Framing and Data Understanding
- [@gustikresna](https://github.com/gustikresna) - Exploratory Data Analysis and Data Preprocessing
- [@elvanselvano](https://github.com/kylelobo) - Modeling, Evaluation, and Deployment

Special thanks to our mentor, [@Muhammad](https://github.com/M46F) who provided a lot of feedback and insights.

## 🎉 Acknowledgments <a name = "acknowledgments"></a>

This project uses an open source dataset, which include 48 explanatory features and 158,957 entries of housing sales in Washington, D.C from 1947 to 2018. The data is available at [Open Data D.C.](https://opendata.dc.gov/) and the residential and address point data is managed by the [Office of the Chief Technology Officer](https://octo.dc.gov/).
