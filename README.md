# Research-Methodology-Stock-Market-Price-Prediction (v1.0)
![GitHub followers](https://img.shields.io/github/followers/nishitshahnz?style=social)
![Profile views](https://gpvc.arturio.dev/nishitshahnz)

In this project I have developed a system that helps users predict long term and short term stock prices.

## Description
The main objective of this project is to help users who are new to stock market to predict stock price. Generally it is seen that most new comers follow multiple trading gurus and follow their tips and yet loose their entire capital. So to avoid this thing and help them learn about how market works and how prices are predicted I am using various Machine Leraning Models to predict short term stock prices. And for long term stock price prediction I am usnig Long Short Term Memory concept. After usign both this method I am successfully able to achieve nice accuracy with some limitation which are mentioned below.


## Flow For The Project

![methodology](https://user-images.githubusercontent.com/42907233/129965202-71f1d11b-dbcb-4f66-9803-1d59a4aa0ef9.PNG)


## Getting Started

### Dependencies
* Software
  * Anaconda Navigator
* Libraries
  * Keras
  * Matplotlib
  * Pandas
  * Numpy
  * Tensorflow
  * Sklearn
* Hardware specifications
  * RAM : 2 to 8 GB
  * Disk space : few KB's
* Language used
  * Python : version 3.0 and above
  
* **Highly Recommend use of google colab to avoid any dependency related issues because it already provides all the dependecies mentined above**

### Dataset
* I am using one of the dataset from Nifty 50 Stock Market Data. The data is from year 2000 to 2021 which is actually pretty big so make sure you trim it down to last 5 years data using MS Excel, otherwise simply use the one dataset file which is available with project files or you can even mail me on the email id provided below to get any help needed with pre-processing or stuff like that. 
* Link to download the dataset: https://www.kaggle.com/rohanrao/nifty50-stock-market-data

### Manifest
```
All_Models_Without_LSTM.ipynb / All_Models_Without_LSTM.py :

  * Code for spliiting the data into training and testing data.
  * Code for All Machine Learning Models such as Linear Regression, Random Forest Regressor, k-Nearest Neighbour Regressor, Support Vector Regressor, Linear Support Vector Regressor, Decision Tree Regressor
  * Code that shows accuracy of each individual model along with their predicted output for user provided input.

LSTM.ipynb / LSTM.py :

  * Code for splitting the data into training and testing data.
  * Code for training the data onto training set and testing on testing set.
  * Code to show the price prediction of future 30 days using Matplotlib.
  
```

### Executing program

Download dataset from above given link or download the zip file of entire project.</br>

* **Highly Recommend  to download zip file of entire project and extract dataset from their because if you download the dataset from kaggle website, pre-procesing of data has to be done manually using excel**

*If using local machine you can either download software mentioned (Anaconda Navigator) above and then download any missing dependencies or if you are using other IDE of your choice and then download all mentioned dependencies.*</br>
*If the running environment is google colab just make sure to upload the extracted folder which you have received after extracting the zip file.*

Method 1 (For colab):
* Download and run All_Models_Without_LSTM.ipynb to check stock price prediction for one day, or you can even run LSTM.ipynb to check stock price prediction for next 30 days.

Method 2 (For local machine):
* If you have downloaded Anaconda Navigator, open Jupyter Notebook first, it will open a link in your default browser.
* Make sure you upload the files on that default path.
* Once uploaded you can see those in browser and can run them from there.
* Secondly if you are using any IDE of your choice then make sure you change all the paths given in code according to your local machine

### Results
![single_day_price_prediction_using_ml_models](https://user-images.githubusercontent.com/42907233/129986932-5d8e393c-3499-4915-8efa-e2e4e529680c.PNG)

The above table is made by running the All_Models_Without_LSTM.ipynb file and the results have been kept in a table for better understanding of user.

![long_term_price_prediction_using_LSTM_along_with_trading_view](https://user-images.githubusercontent.com/42907233/129986928-241cbf72-8372-490d-9e2c-e6f135a450a5.PNG)

The above graph is from trading view website where I have tried checking whether the output delivered by LSTM was accurate or not and it can be clearly seen that the curve matches the output from LSTM.ipynb file. Just to make it easier for user to understand I have drawn same curve on trading view to show our predicted output to actual output.

## Limitations
Stock Market comprises of many other features such as Sentiment Analysis of Market, specific news release of a Stock and much more. Including all these in this project was not feasible because of two reasons first being the computation power and second as it was not economically feasible.
This system is for users who have a little knowledge about the market and I **Highly Recommend not to take any trades using this system as it still has some limitations which can be further solved with time and money**. If any trades are taken using this system then the user using this system will solely be responsible for any profit or loss (i.e. I (Nishit Manishbhai Shah) dont take any responsibility of trades taken using this system.

## Authors
Nishit Manishbhai Shah<br/>
shahnishit48@gmail.com

## Version History
* 1.0
    * Final Release

## License
This project is not licensed.

## Project status
The project is complete</br>
But I still believe, a lot more can be done with high computation power and as we are seeing advancement in the field of Natural Language Processing soon we will be able to tackle the issue of sentiment in market.
