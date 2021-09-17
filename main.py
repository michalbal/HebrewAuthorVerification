
import WorksRetrival
# import parser
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, \
    roc_curve, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.model_selection import train_test_split
import pickle
from embedder import Embedder

from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier().fit(X_train_tf, train_df.label)
from sklearn.linear_model import LogisticRegression
# clf=LogisticRegression().fit(X_train_tf, train_df.label)

START_FROM = 10

def create_svm_model_and_split_data(samples_df, author_name):
    encodings = samples_df.apply(parser.create_document_feature_vector, axis=1).to_numpy()
    # encodings = [parser.create_document_feature_vector(line) for line in samples_df]
    X_train, X_test, y_train, y_test = train_test_split(
        encodings, samples_df['Label'], test_size=0.2, random_state=42)

    svm_model_path = ".\models\\" + author_name + "_svm_model.pkl"
    if not os.path.exists(svm_model_path):
        svm_model = SGDClassifier()
        svm_model.fit(X_train, y_train)
        save_model(svm_model, svm_model_path)
    else:
        svm_model = load_model(svm_model_path)
    show_model_success_on_dataset(svm_model, svm_model_path, X_test, y_test, author_name)

    # Random forest classifier
    random_forest_model_path = ".\models\\" + author_name + "_random_forest_model.pkl"
    if not os.path.exists(random_forest_model_path):
        random_forest_model = RandomForestClassifier()
        random_forest_model.fit(X_train, y_train)
        save_model(random_forest_model, random_forest_model_path)
    else:
        random_forest_model = load_model(random_forest_model_path)
    show_model_success_on_dataset(random_forest_model, random_forest_model_path, X_test, y_test,
                                  author_name + " random_forest")

    return svm_model, X_train, X_test, y_train, y_test


def save_model(model, path):
    with open(path, "wb") as file:
        pickle.dump(model, file)


def load_model(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def show_model_success_on_dataset(model, model_name, x_test, y_test, author_name):
    y_predicted = model.predict(x_test)
    print(model_name, " accuracy: ",
          accuracy_score(y_test, y_predicted)," and roc: ", roc_auc_score(y_test, y_predicted), " of author ", author_name)
    print(classification_report(y_test, y_predicted))




if __name__ == '__main__':
    # i = START_FROM
    # for num in range(START_FROM, 27779):
    #     try:
    #         i += 1
    #         WorksRetrival.get_work(num)
    #     except:
    #         print("Exception! Reached num ", i)
    #
    # WorksRetrival.clean_directory("./authors")

    # attempt = pd.read_csv("authors/דוד פרישמן/בחירות.csv")
    # text = attempt["Text"][0]
    # print("text is ", text)
    parser = Embedder(4)

    # Train models for all authors
    dirs = os.listdir("authors")
    all_files_paths = []
    for author_dir in dirs:
        author_path = "authors/" + author_dir
        samples_df = WorksRetrival.retrieve_samples_for_author(author_path,
                                                               author_dir)

        svm_model, X_train, X_test, y_train, y_test = create_svm_model_and_split_data(
            samples_df, author_dir)

