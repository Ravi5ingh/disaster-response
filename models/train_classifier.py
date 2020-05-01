import sys
import models.pipelines.sparsevec_randomforest_pipeline as sp
import utility.util as ut

def main():
    """
    Entry point
    """

    if len(sys.argv) == 4:
        database_filepath, table_name, model_filepath = sys.argv[1:]
        
        print('Create and Fit Pipeline...')
        model = sp.SparseVecRandomForestPipeline(database_filepath, 'Disaster')
        model.init_fit_eval()

        print('Saving pipeline...\n    PIPELINE: {}'.format(model_filepath))
        ut.to_pkl(model, model_filepath)
        print('Trained pipeline saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument, the table name as the second argument, and'
              ' the filepath of the pickle file to save the model to as the third'
              ' argument. \n\nExample: python -m models.train_classifier '
              'data/DisasterResponse.db Disaster models/model.pkl')


if __name__ == '__main__':
    main()