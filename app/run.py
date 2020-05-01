import json as js
import plotly as pl
import flask as fl
import utility.util as ut


app = fl.Flask(__name__)

# Initialize file names
disaster_db_filename = __file__[0:__file__.rindex('\\')] + '\\..\\data\\DisasterResponse.db'
disaster_table_name = 'Disaster'
model_filename = __file__[0:__file__.rindex('\\')] + '\\..\\models\\model.pkl'

# load data
disaster_df = ut.read_db(disaster_db_filename, disaster_table_name)

# load model
model = ut.read_pkl(model_filename)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = disaster_df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    graphs = [
        dict(
            data=[
                dict(
                    x=genre_names,
                    y=genre_counts,
                    type='bar'
                )
            ],
            layout=dict(
                title='Distribution of Message Genres'
            )
        )
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = js.dumps(graphs, cls=pl.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return fl.render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = fl.request.args.get('query', '')

    # use model to predict classification for query
    classification_results = model.predict(query)

    # This will render the go.html Please see that file. 
    return fl.render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()