import streamlit as st

from src.renders import render_classify_section, render_thematic_section, \
    render_training_section, render_ner_section


def main():
    st.set_page_config(layout="wide")
    st.sidebar.title("Разделы")
    page = st.sidebar.radio("Выбрать раздел:", ["Классификация документов",
                                                 "Обучение классификатора",
                                                 "Тематическое моделирование",
                                                "Распознавание именных сущностей (NER)"])

    st.title(page)
    if page == "Классификация документов":
        render_classify_section()

    elif page == "Обучение классификатора":
        render_training_section()  
    
    elif page == "Тематическое моделирование":
        render_thematic_section()

    elif page == "Распознавание именных сущностей (NER)":
        render_ner_section()


st.session_state['uploaded_files'] = []
st.session_state['training_file'] = None
st.session_state['ner_file'] = None

if __name__ == "__main__":
    main()