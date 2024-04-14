import streamlit as st
import pandas as pd
import requests

from src.constants import SERVER_API, NER_ENTITIES, NER_COLUMNS


def render_ner_section():
    """
    Renders the NER section of the Streamlit app.

    This function finds entities in certain text file
    """
    st.write('''
    Данный раздел позволяет вычленять именные сущности из текста. Такие как организации, имена людей и местоположения.
    Это очень полезно, чтобы узнать более подробную информацию о документе и сделать выводы о принадлежности тому или иному классу.
     ''')
    uploaded_file = st.file_uploader("Загрузка датасета",
                                     accept_multiple_files=False,
                                     key="ner_loading")

    if uploaded_file is not None:
        st.session_state['ner_file'] = uploaded_file

        file = [('file', (st.session_state['ner_file'].name,
                          st.session_state['ner_file'].read(),
                          'multipart/form-data'))]
        st.info("Пожалуйста, подождите!")
        response = requests.request("POST", f"{SERVER_API}/namedEntityRecognize", files=file)
        result = response.json()

        if len(result) == 0:
            st.error("В данном файле отсутствуют именные сущности!")
            return

        df = pd.DataFrame(result)
        df = df.rename(columns=NER_COLUMNS)[NER_COLUMNS.values()]
        df["Класс"] = df["Класс"].replace(NER_ENTITIES)
        df.drop_duplicates(inplace=True)

        st.dataframe(df[NER_COLUMNS.values()], use_container_width=True, hide_index=True)
