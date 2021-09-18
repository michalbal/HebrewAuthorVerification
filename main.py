
import WorksRetrival
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, \
    roc_curve, roc_auc_score, classification_report, f1_score
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.model_selection import train_test_split
import pickle
from embedder import Embedder

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

START_FROM = 10


def split_and_encode_data(samples_df):
    encodings = [parser.create_document_feature_vector(line) for index, line in
                 samples_df.iterrows()]
    labels = samples_df['Label'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        encodings,labels , test_size=0.2,
        random_state=42, stratify=labels)

    return X_train, X_test, y_train, y_test


def create_svm_model(X_train, y_train, author_name):
    svm_model_path = ".\models\\" + author_name + "_svm_model.pkl"
    if not os.path.exists(svm_model_path):
        svm_model = SGDClassifier()
        svm_model.fit(X_train, y_train)
        save_model(svm_model, svm_model_path)
    else:
        svm_model = load_model(svm_model_path)

    return svm_model


def create_random_forest_model(X_train, y_train, author_name):
    random_forest_model_path = ".\models\\" + author_name + "_random_forest_model.pkl"
    if not os.path.exists(random_forest_model_path):
        random_forest_model = RandomForestClassifier(n_estimators=300)
        random_forest_model.fit(X_train, y_train)
        save_model(random_forest_model, random_forest_model_path)
    else:
        random_forest_model = load_model(random_forest_model_path)
    return random_forest_model


def create_nearest_neighbors_model(X_train, y_train, author_name, num_neighbors):
    nearest_neighbors_model_path = ".\models\\" + author_name + "_nearest_" + str(num_neighbors) +"_neighbors_model.pkl"
    if not os.path.exists(nearest_neighbors_model_path):
        nearest_neighbors_model = KNeighborsClassifier(n_neighbors=num_neighbors)
        nearest_neighbors_model.fit(X_train, y_train)
        save_model(nearest_neighbors_model, nearest_neighbors_model_path)
    else:
        nearest_neighbors_model = load_model(nearest_neighbors_model_path)
    return nearest_neighbors_model


def create_decision_tree_model(X_train, y_train, author_name):
    decision_tree_model_path = ".\models\\" + author_name + "decision_tree_model.pkl"
    if not os.path.exists(decision_tree_model_path):
        decision_tree_model = DecisionTreeClassifier()
        decision_tree_model.fit(X_train, y_train)
        save_model(decision_tree_model, decision_tree_model_path)
    else:
        decision_tree_model = load_model(decision_tree_model_path)
    return decision_tree_model


def save_model(model, path):
    with open(path, "wb") as file:
        pickle.dump(model, file)


def load_model(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def show_model_success(model, model_name, x_test, y_test, author_name):
    y_predicted = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_predicted)
    auc_score = roc_auc_score(y_test, y_predicted, average='weighted', labels=np.unique(y_predicted))
    f1_score_model = f1_score(y_test, y_predicted, average='weighted', labels=np.unique(y_predicted))
    print(model_name, " Accuracy: ",
          accuracy," and AUC: ", auc_score, " F1 score: ", f1_score_model, " of Author ", author_name)
    # print(classification_report(y_test, y_predicted))
    return accuracy, auc_score, f1_score_model


def plot_random_forest_results():
    scores_random_forest_df = pd.read_csv("./scores.csv", encoding="utf-8-sig")

    plt.plot('Num_Samples', 'Accuracy', 'bo', data=scores_random_forest_df)
    plt.xlabel('Num Samples')
    plt.ylabel('Accuracy')
    plt.xlim(0, 400)
    plt.show()

    plt.plot('Num_Samples', 'AUC', 'ro', data=scores_random_forest_df)
    plt.xlabel('Num Samples')
    plt.ylabel('AUC Score')
    plt.xlim(0, 400)
    plt.show()

    plt.plot('Num_Samples', 'F1', 'go', data=scores_random_forest_df)
    plt.xlabel('Num Samples')
    plt.ylabel('F1 Score')
    plt.xlim(0, 400)
    plt.show()



if __name__ == '__main__':
    # Get files from Ben Yehuda
    # i = START_FROM
    # for num in range(START_FROM, 27779):
    #     try:
    #         i += 1
    #         WorksRetrival.get_work(num)
    #     except:
    #         print("Exception! Reached num ", i)
    #
    # WorksRetrival.clean_directory("./authors")

    # Train models for all authors
    parser = Embedder(2)
    dirs = os.listdir("authors")
    num_of_authors = len(dirs)

    accuracy_svm_sum = 0
    roc_score_svm_sum = 0
    f1_score_svm_sum = 0

    accuracy_forest_sum = 0
    roc_score_forest_sum = 0
    f1_score_forest_sum = 0

    accuracy_decision_tree_sum = 0
    roc_score_decision_tree_sum = 0
    f1_score_decision_tree_sum = 0

    all_files_paths = []
    for author_dir in dirs:
        author_path = "authors/" + author_dir
        samples_df = WorksRetrival.retrieve_samples_for_author(author_path,
                                                               author_dir)

        X_train, X_test, y_train, y_test = split_and_encode_data(samples_df)

        svm_model = create_svm_model(X_train, y_train, author_dir)
        accuracy, auc_score, f1_score_model = show_model_success(svm_model, "svm_" + author_dir, X_test, y_test, author_dir)
        accuracy_svm_sum += accuracy
        roc_score_svm_sum += auc_score
        f1_score_svm_sum += f1_score_model

        random_forest_model = create_random_forest_model(X_train, y_train, author_dir)
        accuracy, auc_score, f1_score_model = show_model_success(random_forest_model, "random_forest_" + author_dir, X_test, y_test, author_dir)
        accuracy_forest_sum += accuracy
        roc_score_forest_sum += auc_score
        f1_score_forest_sum += f1_score_model

        decision_tree_model = create_decision_tree_model(X_train, y_train,
                                                         author_dir)
        accuracy, auc_score, f1_score_model = show_model_success(
            decision_tree_model, "decision_tree_" + author_dir, X_test, y_test,
            author_dir)
        accuracy_decision_tree_sum += accuracy
        roc_score_decision_tree_sum += auc_score
        f1_score_decision_tree_sum += f1_score_model

    print("Num of authors is: ", num_of_authors)
    print(" Average results are: ")
    avg_svm_accuracy = accuracy_svm_sum / num_of_authors
    avg_svm_auc = roc_score_svm_sum / num_of_authors
    avg_svm_f1 = f1_score_svm_sum / num_of_authors
    print(" Average SVM: \n accuracy: ", avg_svm_accuracy, " Auc: ",avg_svm_auc, " F1: ",  avg_svm_f1)

    avg_forest_accuracy = accuracy_forest_sum / num_of_authors
    avg_forest_auc = roc_score_forest_sum / num_of_authors
    avg_forest_f1 = f1_score_forest_sum / num_of_authors
    print(" Average Random Forest: \n accuracy: ", avg_forest_accuracy, " Auc: ",
          avg_forest_auc, " F1: ", avg_forest_f1)

    avg_decision_tree_accuracy = accuracy_decision_tree_sum / num_of_authors
    avg_decision_tree_auc = roc_score_decision_tree_sum / num_of_authors
    avg_decision_tree_f1 = f1_score_decision_tree_sum / num_of_authors
    print(" Average Decision Tree: \n accuracy: ", avg_decision_tree_accuracy,
          " Auc: ",
          avg_decision_tree_auc, " F1: ", avg_decision_tree_f1)

