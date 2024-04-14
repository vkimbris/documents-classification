import streamlit as st
import requests
import pandas as pd

from src.constants import SERVER_API


def render_thematic_section():
    """
    Renders the thematic section of the Streamlit app.

    This function handles the user input for the number of topics and random seed,
    performs the topic extraction from the training set.
    """
    st.write('''
   Найдите в документах скрытые темы, используя этот сервис.
     ''')
    topics = st.number_input('Введите количество топиков', min_value=5, max_value=100, value=5, step=1)
    seed = st.number_input('Введите random seed', min_value=5, value=42, step=1)

    if st.button("Вывести топики"):
        st.success(f'Пожалуйста, подождите!')

        response = requests.request("POST", 
                                    f"{SERVER_API}/topicModeling?n_topics={topics}&seed={seed}")
        topics_json = response.json()
        topics = [topic['topicKeyWords'] for topic in topics_json]

        df = pd.DataFrame(topics)
        df.index = [f"Топик {i+1}" for i in range(len(topics))]
        df.index.name = 'Топики'
        df.columns = [f"Ключевое слово {i+1}" for i in range(len(df.columns))]

        st.table(df)