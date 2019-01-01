# Disaster Response Pipeline Project

### Table of Contents
1. [Installation](https://github.com/ddh4/disaster-response-project#installation)
2. [About](https://github.com/ddh4/disaster-response-project#about)
3. [File Descriptions](https://github.com/ddh4/disaster-response-project#file-descriptions)
4. [Instructions](https://github.com/ddh4/disaster-response-project#instructions)
5. [Licensing, Authors, and Acknowledgements](https://github.com/ddh4/disaster-response-project#licensing-authors-and-acknowledgements)

### Installation
The libraries necessary are:
- NumPy
- Scikit Learn
- Pandas
- Plotly
- SqlAlchemy
- NLTK
- Pickle
- Flask

All libraries necessary are bundled with Anaconda distribution of Python 3.*.. 

### About
The final output of this project is a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

![app_home](https://user-images.githubusercontent.com/39163121/50575732-9fc5bf00-0dfb-11e9-9f64-af2df01b414b.png)

The project is comprised of three core components:
1. ETL Pipeline: Loading the categories.csv and messages.csv datasets, merging, cleaning and
saving the output in an SQLite database.
2. ML Pipeline: Loading the data from the SQLite database we created in the ETL pipeline, splitting for training and testing, building a text processing and machine learning pipeline, training the model with GridSearchCV, output evaluation metrics and export the final model as a pickle file.
3. Flask Web App: Setup a Flask web app that shows some visualisations of the dataset as well as allowing the user to input a message which will be classified by the trained model. The classification labels are output in green and remaining labels are output in grey.

![app_predict](https://user-images.githubusercontent.com/39163121/50575738-c126ab00-0dfb-11e9-869a-8d76a3bdd700.png)

### File Descriptions

The app folder contains the files neccessary to run the Flask application, loading using run.py which renders the html templates.

The data folder contains the files neccessary to generate the SQLite database and app visualisation, using the categories.csv and messages.csv datasets.

The models folder contains the files neccessary to train a multi-output classifier to predict messages for disaster response. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To produce a dataset for visualisation in the app
        `python data/process_data_visualisations.py`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:5000/

### Licensing, Authors, and Acknowledgements
Thanks to Udacity and Figure Eight for providing the datasets and posing the project.
The code in this repository can be used freely.

