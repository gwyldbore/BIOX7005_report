import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle

import joblib



def main():    

    model_path = snakemake.input.model
    test_data_path = snakemake.input.embedding_df
    output_path = snakemake.output.logreg_results

    # load the trained model and label encoder
    logistic_model, label_encoder = joblib.load(model_path)

    # test_data = pd.read_csv(test_data_path)
    with open(test_data_path, "rb") as input_file:
        test_data = pickle.load(input_file)

    info_column = test_data['info']
    # test_data.columns = test_data.columns.astype(str)
    X_test = test_data.drop(columns=['info', 'sequence', 'model_name', 
                                 'protbert_max_embedding', 'protbert_mean_embedding', 
                                 'protbert_weighted_embedding', 'num_mutation'])
    
    
    
    # Transform 'protbert_mean_embedding' into separate features (as in training)
    X_test = X_test.join(pd.DataFrame(X_test.pop('protbert_cls_embedding').tolist(), index=X_test.index))

    X_test.columns = X_test.columns.astype(str)


    y_pred_encoded = logistic_model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    y_proba = logistic_model.predict_proba(X_test)

    proba_columns = [f"Proba_{cls}" for cls in label_encoder.classes_]
    proba_df = pd.DataFrame(y_proba, columns=proba_columns)



    # Combine the 'info' column, predictions, and probabilities into one DataFrame
    results_df = pd.DataFrame({
        'info': info_column,
        'embedding_prediction': y_pred
    })
    results_df = pd.concat([results_df, proba_df], axis=1)

    results_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()