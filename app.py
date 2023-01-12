import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
import platform

# repair error
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

# title
st.title('Classification Model')

st.header('Classes: Person | Book | Fruit')

# input image
file = st.file_uploader('Upload image', type=['png','jpeg','jpg','jfif','svg'])
if file:
    st.image(file)

    img = PILImage.create(file)

    # read model
    model = load_learner('different_model.pkl')

    # predict
    pred, pred_id, probs = model.predict(img)

    st.success(f'Predicted: {pred}')
    st.info(f'Probability: {probs[pred_id]*100:.1f}%')

    # plotting
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    px.xlabel("prediction")
    px.ylabel("class")
    st.plotly_chart(fig)
