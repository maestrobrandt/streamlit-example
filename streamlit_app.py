from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st




import streamlit as st
import altair
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")



import numpy as np
import pandas as np



import pickle

forest = pickle.load(open( "forest.p", "rb" ))


st.title('Admission Predictor')
st.subheader('GPA:')
gpa_slider = st.slider("Choose to the nearest 0.1", min_value = 0.0, max_value = 4.0, step = 0.1)
st.write('GPA:', gpa_slider)
gpa = gpa_slider
st.subheader('GRE Score:')
gre_slider = st.slider("Choose to the nearest 1", min_value = 0.0, max_value = 800.0, step = 1.0)
st.write('GRE Score:', gre_slider)
gre = gre_slider
tn_box = st.checkbox('Attended Top Notch')
tn = 0
if not tn_box:
    st.write('Did not attend top notch school')
else:
    st.write('Attended top notch school')
    tn = 1


# In[6]:


st.header('Chance of Admission:')
button = st.button('PREDICT')
if button:
    st.write(round(((forest.predict_proba([[gre, tn, gpa]])[:,1])[0])*100, 2), "%")
