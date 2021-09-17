import shutil

import requests
from bs4 import BeautifulSoup
from time import sleep
import pandas as pd
from pathlib import Path
import os
import random


def get_work(work_number: int):
    url = "https://benyehuda.org/read/" + work_number.__str__()
    request = requests.get(url, verify=False)
    if request.status_code != 200:
        print("work num ", work_number, " is empty")
        return

    txt = request.text
    soup = BeautifulSoup(txt, features="html.parser")

    # Find out if the work is of the correct kind
    breadcrumbs = soup.findAll(id='breadcrumbs')
    if len(breadcrumbs) == 0:
        print("Work num ", work_number, " does not have breadcrumb")
        return

    type_info : str = breadcrumbs[0].text

    if type_info.find("שירה") != -1 or type_info.find("מחזות") != -1 or type_info.find("משלים") != -1:
        # Perhaps we'll want to reconsider משלים
        return

    title_info = soup.findAll(class_="headline-1-v02 work-name-top")
    if len(title_info) == 0:
        print("Could not find title of work num ", work_number)
        return

    title = title_info[0].text

    text_info = soup.findAll(id='actualtext')
    if len(text_info) == 0:
        print("Could not find text of work num ", work_number)
        return

    text = text_info[0].text

    # Find author
    author_info = soup.findAll(class_="headline-3-v02")
    if len(author_info) == 0:
        print("Could not find author of work num ", work_number)
        return

    author_information : str = author_info[0].text
    splitted = author_information.split("תרגום:")

    author = splitted[0]
    author = author.replace("\n", "")
    author = author.replace("מאת:", "").strip()

    translator = ""
    language = ""
    if len(splitted) > 1:
        # This work is translated
        translation_info = splitted[1].replace("\n", "").split("(")
        translator = translation_info[0].strip()
        language = translation_info[1].replace(")", "").strip()
        if author == "אלמוני/ת":
            author = translator

        df_data = {'Title': [title], 'Translator': [translator],
                   'Lang': [language], 'Author': [author], "Text": [text]}
        df = pd.DataFrame.from_dict(df_data)

        # Create file for the work, and directory for author and translator if needed
        path = Path("./authors/"+author+"/"+translator+"/"+title)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv("./authors/"+author+"/"+translator+"/"+title+".csv", encoding="utf-8-sig")

        # Create file for the translator as well
        path = Path("./translators/" + translator + "/" + author + "/" + title)
        path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(
            "./translators/" + translator + "/" + author + "/" + title + ".csv",
            encoding="utf-8-sig")
    else:
        # Create file for the work, and directory for author if needed
        df_data = {'Title': [title], 'Translator': [translator],
                   'Lang': [language], 'Author': [author], "Text": [text]}
        df = pd.DataFrame.from_dict(df_data)

        path = Path("./authors/" + author + "/" + title)
        path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(
            "./authors/" + author + "/" + title + ".csv", encoding="utf-8-sig")


def clean_directory(dir_path):
    min_num_files_keep = 10
    dirs = os.listdir(dir_path)
    all_files_paths = []
    for author_dir in dirs:
        author_path = dir_path + "/" + author_dir
        contents = os.listdir(author_path)
        file_paths = []
        if len(contents) == 0:
            os.rmdir(author_path)
        else:
            num_files = 0
            for content in contents:
                content_path = author_path + "/" + content
                if os.path.isfile(content_path):
                    if Path(content_path).stat().st_size < 10:
                        # File size <= 0KB
                        os.remove(content_path)
                    else:
                        num_files += 1
                        file_paths.append(content_path)
                else:
                    files = os.listdir(content_path)
                    for file_name in files:
                        file_path = content_path + "/" + file_name
                        if os.path.isfile(file_path):
                            if Path(file_path).stat().st_size < 10:
                                # File size <= 0KB
                                os.remove(file_path)
                            else:
                                num_files += 1
                                file_paths.append(file_path)
                        else:
                            # Directory was created instead of file -
                            # Something went wrong, this should not exist
                            shutil.rmtree(file_path)
            if num_files < min_num_files_keep:
                # We collected too few samples of this author - do not need
                shutil.rmtree(author_path)
            else:
                all_files_paths.extend(file_paths)

    df_data = {'Files': all_files_paths}
    df = pd.DataFrame.from_dict(df_data)
    df.to_csv("./all_files.csv", encoding="utf-8-sig")


def retrieve_samples_for_author(path_to_author_dir: str, author_name: str):
    positive_samples = pd.DataFrame()
    author_dir_content = os.listdir(path_to_author_dir)
    num_of_samples = 0
    for content in author_dir_content:
        content_path = path_to_author_dir + "/" + content
        if os.path.isfile(content_path):
            text = pd.read_csv(content_path)["Text"][0]
            sample_dict = {'Text': text, 'Label': 1}
            positive_samples = positive_samples.append(sample_dict, ignore_index=True)
            num_of_samples += 1
        else:
            # Directory with translator files
            files = os.listdir(content_path)
            for file in files:
                text = pd.read_csv(content_path + "/" + file)["Text"][0]
                sample_dict = {'Text': text, 'Label': 1}
                positive_samples = positive_samples.append(sample_dict, ignore_index=True)
                num_of_samples += 1

    negative_samples = get_different_author_samples_of_size(author_name, num_of_samples)
    return pd.concat([positive_samples, negative_samples], ignore_index=True, axis=0)


def get_different_author_samples_of_size(author: str, size: int):
    files = pd.read_csv("./all_files.csv", encoding="utf-8-sig")['Files']
    samples_df = pd.DataFrame()
    files_chosen = set()
    for i in range(size):
        random_file_path = random.choice(files)
        while author in random_file_path or random_file_path in files_chosen:
            if author in random_file_path:
                print("randomly selected filepath ", random_file_path,
                      " It is a sample of the same author: ", author)
            else:
                print("File ", random_file_path, " was already selected")
            random_file_path = random.choice(files)
        files_chosen.add(random_file_path)
        text = pd.read_csv(random_file_path)["Text"][0]
        sample_dict = {'Text': text, 'Label': 0}
        samples_df = samples_df.append(sample_dict, ignore_index = True)
    return samples_df
