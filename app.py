import streamlit as st
from predict_page4 import show_predict_page
from explore_page2 import show_explore_page
from chat_page import show_chat_page

import streamlit as st
from PIL import Image

def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url(http://placekitten.com/200/200);
                background-repeat: no-repeat;
                padding-top: 120px;
                background-position: 20px 20px;
            }
            [data-testid="stSidebarNav"]::before {
                content: "My Company Name";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

image = Image.open('Miuulpay_preview_rev_1.png')

st.image(image, width =300)

page = st.sidebar.selectbox("Explore Or Predict", ("Predict", "Explore"))
add_logo()
if page == "Predict":
    show_predict_page()
# elif page == "Chat":
#     show_chat_page()
else:
    show_explore_page()



