# Disaster Response Pipeline Project

This is an attempt to use NLP and ML to identify tweets that refer to real disasters. If you follow the steps, then 
what you will end up with is a app running at http://localhost:3001/ that can classify tweets.

This project is currently running here: http://159.65.28.124:3001/

## PyCharm Setup
<ol>
  <li>Clone this repo</li>
  <li>Open PyCharm</li>
  <li>Choose to 'Create new project' in the directory where you have cloned the code</li>
  <li>Make sure you select virtal python environment with a 'venv' folder in the root directory</li>
  <li>NOTE: This code will only work with Python x64. It was built and tested with version 3.7.3</li>
  <li>Say NO to creating with existing source. PyCharm will now create a virtual environment</li>
  <li>Open the PyCharm terminal window. If not already, navigate to the root directory (disaster-response)</li>
  <li>pip install pandas</li> 
  <li>pip install matplotlib</li>
  <li>pip install sqlalchemy</li>
  <li>pip install plotly</li>
  <li>pip install flask</li>
  <li>pip install sklearn</li>
  <li>pip install pipetools</li>
  <li>pip install gensim</li>
  <li>pip install seaborn</li>
  <li>pip install nltk</li>
  <li>python -m nltk.downloader -d venv/nltk_data all</li>
</ol>

python -m data.process_data data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db Disaster
python -m models.train_classifier data/DisasterResponse.db Disaster models/model.pkl
python -m app.run

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        
        `python -m data.process_data data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db Disaster`
    - To run ML pipeline that trains classifier and saves
        
        `python -m models.train_classifier data/DisasterResponse.db Disaster models/model.pkl`

2. Run the following command in the app's directory to run your web app.
    
    `python -m app.run`

3. Go to http://localhost:3001/

## IMPORTANT NOTES
* The model vectorizes messages using word2vec. There is a word2vec cache in the file models/nl/word2vec_cache.pkl
* At any point if, the code encounters a word that is not in the cache, it will ATTEMPT TO DOWNLOAD Google's trained model.
* Everything should just work without user intervention but in case it doesn't, the model can be found here: (https://drive.google.com/uc?export=download&confirm=wNia&id=1kzCpXqZ_EILFAfK4G96QZBrjtezxjMiO)
* It needs to be here: models/nl/GoogleWord2VecModel.bin
* Justification: The reason I chose Google's model instead of training my own word2vec model is that I wanted to capture the 'true'
meaning of the words in a global context rather than a confined context of the training data. This would be reflected in the fact that words associated with catastrophes are closer to each other than regular words.