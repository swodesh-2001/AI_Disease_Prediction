# flask app
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

DATASET_URL = "dataset.csv"  
df = pd.read_csv(DATASET_URL)
df['Disease'] = df['Disease'].map(lambda x: x.strip())
symptoms = df[df.columns[1:]].values.flatten()
cleaned_symptoms = [x.strip() for x in symptoms if str(x) != 'nan']
unique_symptoms = list(set(cleaned_symptoms))

rndm_frst_clf = RandomForestClassifier(n_estimators = 100)

def cleanup_dataframe(df):
    (df_numrows, df_numcols) = df.shape
    col_names = df[df.columns[1:]].values.flatten()
    cleaned_col_names = [x.strip() for x in col_names if str(x) != 'nan']
    unq_col_names = list(set(cleaned_col_names))
    cleaned_dataframe = pd.DataFrame(columns=unq_col_names)
    for x in range(0,df_numrows):
        cur_symp = df.iloc[x].values[1:]
        cur_cleaned_symp = [x.strip() for x in cur_symp if str(x) != 'nan']
        col_val = np.zeros(shape=(len(unique_symptoms),),dtype=float)
        for y in cur_cleaned_symp:
            col_index = unique_symptoms.index(y)
            col_val[col_index] = 1
        cleaned_dataframe.loc[len(cleaned_dataframe)] = col_val
    return cleaned_dataframe

def get_train_test_df(y,cleaned_dataframe,test_size=0.3,random_state=100):
    X = cleaned_dataframe.values
    y = y
    return train_test_split(
        X,
        y,
        stratify=y,
        test_size=test_size,
        random_state=random_state
    )

def vote_of_majority(symptoms):
    return rndm_frst_clf.predict([symptoms])

def train_dataset(classifier,X_train, X_test, y_train, y_test):
    classifier.fit(X_train,y_train)
    pred = classifier.predict(X_test)
    return "Accuracy:"+ str(metrics.accuracy_score(y_test, pred))

cln_df = cleanup_dataframe(df)
y = df['Disease'].values
X_train, X_test, y_train, y_test = get_train_test_df(y,cln_df)

@app.route("/check_symptom",methods=["POST","GET"])
def check_disease():
    if request.method == 'POST':
        hidden_symptoms = request.form['hidden_symptom']
        hidden_symptoms = hidden_symptoms.split(',')
        hidden_symptoms = [x.strip() for x in hidden_symptoms]
        val = np.zeros(shape=(len(unique_symptoms),),dtype=float)
        for y in hidden_symptoms:
            index = unique_symptoms.index(y)
            val[index] = 1
        prediction = vote_of_majority(val)
        return prediction[0]

@app.route('/')
def index():
    return render_template('index.html',Symptoms=unique_symptoms)


if __name__ == '__main__':
    train_dataset(rndm_frst_clf, X_train, X_test, y_train, y_test)
    app.run(debug=True)
