import streamlit as st
from io import BytesIO
import pandas as pd
import requests

from src.constants import EXAMPLE_TRAIN_TABLE, SERVER_API
from src.utils import create_classification_report


def render_training_section() -> None:
    """
    Renders the training section of the Streamlit app.
    
    This function handles dataset file upload and starts model training if appropriate.
    """    
    # st.header("Загузите тренировочный датасет")
    st.write('''Тренировочный датасет будет объединён с нашим датасетом и модель будет полностью переобучена и заменена в программе.
             Также вы можете добавлять документы с новыми классами и наша модель адапатируется под ваши данные.

             Пример csv файла, который должен быть загружен:
    ''')
    st.markdown(EXAMPLE_TRAIN_TABLE, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Загрузка датасета",
                                      accept_multiple_files=False,
                                      key="training_uploader", type="csv")

    if uploaded_file is not None:
    
        st.session_state['training_file'] = uploaded_file

        try:
            df = pd.read_csv(BytesIO(uploaded_file.getvalue()))
        except pd.errors.EmptyDataError:
            st.error("Файл пуст или не соответствует заданному формату!", icon="🚨")
            return

        if df.columns.to_list() != ["text", "class"]:
            st.error("Файл не соответствует необходимому формату!", icon="🚨")
            return

        if st.button("Начать обучение"):            
            st.success("Обучение началось... Пожалуйста, подождите!")
            file = [('train_data', (st.session_state['training_file'].name,
                                    st.session_state['training_file'].read(),
                                    'text/csv'))]
            
            response = requests.request("POST", f"{SERVER_API}/updateModel", files=file).json()

            if not response.get("status"):
                st.error("Не удалось обучить модель!", icon="🚨")
                return
            
            st.success("Обучение прошло успешно, модель обновилась!")

            st.markdown(create_classification_report(response["classificationReport"]))