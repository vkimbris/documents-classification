import streamlit as st
import requests
from collections import Counter
from typing import List, Dict
import json

from src.utils import create_dataframe, get_labels, get_status
from src.constants import SERVER_API


def classify_and_show_result(file_labels: List[str], result: List[Dict[str, str]]) -> None:
    """
    Classifies the uploaded files and displays the results in a table.

    Parameters
    ----------
    file_labels : List[str]
        A list of labels assigned to the uploaded files.
    result: List[Dict[str, str]]
        A list with class information for every loaded document.
    """
    available_labels = get_labels()
    count_labels = dict(Counter(file_labels)) 
    bool_labels = {label: get_status(True) if label in file_labels else get_status(False) 
              for label in available_labels}
    
    df = create_dataframe(count_labels, bool_labels)
    
    st.header("Классы документов в загрузке")
    if file_labels:
        st.dataframe(df.style.set_properties(**{"vertical-align": "text-top"}), 
                     use_container_width=True, hide_index=False, height=None)

    st.header("Классы по каждому документу")
    json_result = json.dumps({"result": result}, ensure_ascii=False)
    st.json(json_result, expanded=True)
    st.download_button(
            label="Загрузить результат",
            file_name="data.json",
            mime="application/json",
            data=json_result,
                    )


def render_classify_section():
    """
    Renders the document classification section of the Streamlit app.
    
    This function handles document files upload and performs classification if appropriate.
    """
    st.write('''
    Классифицируйте свои документы по классам, загрузите документы любого текстового формата (docx, pdf, rtf, и т.д.) и получите
             таблицу с результатами классификации. Также получите возможность скачать JSON файл с информацией
             о том, какой класс принадлежит конкретному документу.
     ''')

    files_to_send = []
    uploaded_files = st.file_uploader('Загрузка документов', accept_multiple_files=True)
    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.read()
            if len(file_bytes) == 0:
                continue
            files_to_send.append(("files", (uploaded_file.name, file_bytes, "multipart/form-data")))
            st.session_state['uploaded_files'].append(uploaded_file)

        if files_to_send:
            response = requests.request("POST", f"{SERVER_API}/classifyDocuments", files=files_to_send)
            result = response.json()
            file_labels = [file["label"] for file in result]
            classify_and_show_result(file_labels, result)
