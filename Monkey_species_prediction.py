import json
import sys
from fastcore.all import *
import pathlib, pathlib._local
import streamlit as st
import traceback

try:
    from fastai.vision.all import *
except Exception:
    st.code(traceback.format_exc())
    st.stop()

@st.cache_resource
def load_model():
    return load_learner('export_monkey.pkl')

@st.cache_data
def load_info():
    with open('monkeys.json', encoding = 'utf-8') as f:
        return json.load(f)

if sys.platform == 'win32':
    pathlib.PosixPath = pathlib.WindowsPath
    pathlib._local.PosixPath = pathlib._local.WindowsPath

Uploaded_image = st.file_uploader('Upload a image', type = ['jpg', 'jpeg', 'png'])

if Uploaded_image is not None:
    image = PILImage.create(Uploaded_image.getvalue())
    st.image(image, caption = 'Your uploaded image')
    learn_inf = load_model()
    info = load_info()
    pred, pred_indx, prob = learn_inf.predict(image)
    if prob[pred_indx] < 0.6:
        st.warning(f"##### Not sure about this one. Best guess is : {pred} with probability: {prob[pred_indx]:.4f}. This model only knows 8 species.")
    else:
        st.write(f"#### **Prediction:** {pred}; **Probability:** {prob[pred_indx]:.4f}")
    card = info['species'][str(pred)]

    st.write(f"#### **Common Names:** {card['common_names']}")
    st.write(f"**Family:** {card['family']}")
    st.write(f"**Sub-Family:** {card['subfamily']}")
    st.write(f"**IUCN Status:** {card['iucn_status']}")
    st.write(f"**Range:** {card['range']}")
    st.write(f"**Habitat:** {card['habitat']}")
    st.write(f"**Diet:** {card['diet']}")
    st.write(f"**Identifying features:** {card['identifying_features']}")
    st.write(f"**Fun fact:** {card['fun_fact']}")