# Disaster Response Pipeline Project

This is an attempt to use NLP and ML to identify tweets that refer to real disasters. This is very much a work in progress right now...

## PyCharm Setup
<ol>
  <li>Clone this repo</li>
  <li>Open a PyCharm project in the directory where you cloned it</li>
  <li>Say yes to creating with existing source</li>
  <li>In the terminal, run the following commands</li>
  <li>pip install pandas</li> 
  <li>pip install matplotlib</li>
  <li>pip install sqlalchemy</li>
  <li>pip install plotly</li>
  <li>pip install nltk</li>
  <li>pip install flask</li>
  <li>pip install sklearn</li>
  <li>Right click and run the main.py file to auto-create a run config</li>
</ol>

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
