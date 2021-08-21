import shutil

import requests
from bs4 import BeautifulSoup
from time import sleep
import pandas as pd
from pathlib import Path
import os


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
