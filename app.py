import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px

# repair error
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# title
st.title('Classification Model')

# input image
file = st.file_uploader('Rasm yuklash', type=['png','jpeg','jpg','jfif','svg'])
if file:
    st.image(file)

    img = PILImage.create(file)

    # read model
    model = load_learner('different_model.pkl')

    # predict
    pred, pred_id, probs = model.predict(img)

    st.success(f'Bashorat: {pred}')
    st.info(f'Ehtimollik: {probs[pred_id]*100:.1f}%')

    # plotting
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)