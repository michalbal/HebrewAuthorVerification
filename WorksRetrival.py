import requests
from bs4 import BeautifulSoup
from time import sleep
import pandas as pd
from pathlib import Path


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


