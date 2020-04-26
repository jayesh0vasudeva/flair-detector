# Reddit-Flair-Detector
A Reddit Flair Detector web application to detect flairs of India subreddit posts using Machine Learning algorithms. 
The application can be found live at [Reddit Flair Detector](https://shekapp.herokuapp.com).
## Table of Contents

1. [Directory Structure](#directory-structure) 
2. [Installation](#installation)
3. [Project Execution](#project-execution)
4. [Data Aquisition](#data-aquisition)
5. [Flair Classification](#flair-classification)
6. [WebApp](#webapp)
7. [Data Analysis](#data-analysis)
8. [References](#references)
  
### Codebase

The entire code has been developed using Python programming language, utilizing it's powerful text processing and machine learning modules. The application has been developed using Flask web framework and hosted on Heroku web server.

## Installation

The following installation has been tested on Windows 10

This project requires **Python 3** and the following Python libraries installed(plus a few other s depending on task):

- [sklearn](http://scikit-learn.com/)
- [Pytorch](http://pytorch.org/)
- [pandas](pandas.pydata.org/)
- [Numpy](http://numpy.org/)
- [Matplotlib](https://matplotlib.org/) 

## Project Execution

  1. Open the `Terminal`.
  2. Ensure that `Python3` and `pip` is installed on the system.
  3. Create a `virtualenv` by executing the following command: `virtualenv -p python3 env`.
  4. Activate the `env` virtual environment by executing the follwing command: `source env/bin/activate`.
  5. Enter the cloned repository directory and execute `pip install -r requirements.txt`.
  6. Enter the web app folder using cd Reddit-Flair-Detection/webapp
  7. Now, execute the following command: `python app.py` and it will point to the `localhost` with the port.
  8. Hit the `IP Address` on a web browser and use the application.
  
## Data Aquisition

Note-I have included the data in JSON,CSV and BSON formats in the data folder
I had to collect data pertaining to posts on reddit with different flairs 
there were 11 types of flairs-
  
  1. AMA
  2. AskIndia
  3. Business/Finance
  4. Food
  5. Photography
  6. Non-Political
  7. Policy/Economy
  8. Politics
  9. Sports
  10. Science/Technology
  11. Redditquette
  
All other Flair were either treated as being part of one of these or were completely removed
#### For data aquisition ,2 approaches were used -
  1. Parse through `Top` , `Hot` , `New` of `r/india` on reddit and check if one has already parsed through the post ,if not, then append data of that post of the categories -
  
    * Title
    * Flair 
    * Score
    * ID
    * Comment number 
    * User
    * URL
 forming a list for each post and a list of lists as the whole dataset, convert this list into a DataFrame
  
  2. Parse through `r/india` on reddit going through a specific number of posts with a specific Flair and collect data of categories-
  
    * Title
    * Flair 
    * Comments
    * Score
    * ID
    * Comment number 
    * User
    * URL    
save the same in the form of a list for each post and then repeat the same for all flairs
forming a list of list as the whole dataset convert this list into a DataFrame

#### While preparing the final dataset , approach two was used because it provided me with equal number of data elements from each type of flair unlike the first approach , hence giving better results
 so finally 
 
 * Collected 100 India subreddit data for each of the 11 flairs using `praw` module 
 * The data includes *title, comments, body, url, author, score, id, time-created* and *number of comments*.
 * For **comments**, only top level comments are considered in dataset and no sub-comments are present.
## Flair Classification
I basically had three approaches for the problem -

1. Word2vec+DNN/LSTM followed by a DNN
3. TfIdf+RandomForest/Linear SVM/Logistical Regression

#### The First approach 
  1. The ***title*** is cleaned by removing bad symbols and stopwords using `nltk`.
  2. each title is tokenized as a list of clean words that make it up
  3. then a Pretrained Glove word2vec model is used to assign embedding matrix to each word in every title list
  4. Average of all the word matrix in a list of words of a title is taken and this averaged matrix represents the title now and represents the input
  5. The flair with respect to each title is stored seperately and I oneshot this , this represents the output
    
    1. I trained a DNN on this set of Input and Output with 3 dense layers 
    
    2. I reshaped my Input in order to train a LSTM then trained a LSTM followed by a DNN with 2 dense layers
    on this new reshaped set of input and output
#### The Second approach
  1. The ***title*** is cleaned by removing bad symbols and stopwords using `nltk`.
  2. each title is tokenized as a list of clean words that make it up
  3. all the words of a title are joined to form string which is devoid of stopwords and symbols
  4. The titles are then converted into a `Vector` and `TF-IDF` form.
  5. Then, the following ML algorithms (using `scikit-learn` libraries) are applied on them:
    
    a) Random Forest
    b) Linear Support Vector Machine
    c) Logistic Regression
    
    
 ### Results-

| Machine Learning Algorithm | Test Accuracy           |
| -------------------        |:-----------------------:|
| DNN                        | 0.2100                  |
| LSTM                       | 0.4500                  |
| Linear SVM                 | **0.6231343283582089**  |
| Logistic Regression        | 0.6156716417910447      |
| Random Forest              | 0.5410447761194029      |

As you can see the initial results werent satisfactory with Linear SVM getting the best accuracy and NN based models performing extremely bad
So moving forward I dropped the idea of using NN models as my data set was too small for them to properly function

In order to further increase my accuracy I decided to also process the top comments along with the title of a post 
- so I made a new feature that was a combination of strings of title and comments 
- followed the second approach on the same instead of just using titles 

### Results -

| Machine Learning Algorithm | Test Accuracy           |
| -------------------        |:-----------------------:|
| Linear SVM                 | 0.7798507462686567      |
| Logistic Regression        | 0.7910447761194029      |
| Random Forest              | **0.8022388059701493**  |

As you can see the results were much better now and **randomforest** gave me the best accuracy 
So i pickled the randomforest model and used it further on my webapp
## WebApp
The best model - random forest is deployed as a web app. Check the live demo [here](https://shrouded-journey-09442.herokuapp.com/). All the required files can be found [here](https://github.com/jayesh0vasudeva/flair-detector/tree/master/webapp)
## Data Analysis
I performed data analysis on the data that I scraped , but due to some error wasnt able to deploy that as a page on my web app 

**But I have included the code and the result of the same in the [Data analysis folder](https://github.com/jayesh0vasudeva/flair-detector/tree/master/data%20analysis)**

![Imgur](https://i.imgur.com/1U2jEK6.png)

![Imgur](https://i.imgur.com/SBFZ1Hw.png)

![Imgur](https://i.imgur.com/Q6m4mSp.png)

## References
* [Data visualization](https://towardsdatascience.com/data-visualization-using-matplotlib-16f1aae5ce70)
* [LSTM and sentimental analysis](https://towardsdatascience.com/sentiment-analysis-using-lstm-step-by-step-50d074f09948)
* [Scrapping Reddit](https://towardsdatascience.com/scraping-reddit-data-1c0af3040768)
* [Multi-Class Text Classification Model Comparison and Selection](https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568)
* [MongoDB](https://docs.mongodb.com/manual/reference/program/mongodump/)
* [AWD LSTM](https://yashuseth.blog/2018/09/12/awd-lstm-explanation-understanding-language-model/)
