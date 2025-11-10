import streamlit as st
import Orange
import pickle

st.title('Prediksi Bunga Iris (Model Orange3)')

# Load model hasil Save Model (bukan Save Data)
with open('iris.pkcls', 'rb') as f:
    model = pickle.load(f)

# Domain langsung dari model (harus match)
domain = model.domain

# Input pengguna
sepal_length = st.number_input('Sepal Length', min_value=0.0, max_value=10.0, step=0.1, value=5.0)
sepal_width  = st.number_input('Sepal Width',  min_value=0.0, max_value=10.0, step=0.1, value=3.5)
petal_length = st.number_input('Petal Length', min_value=0.0, max_value=10.0, step=0.1, value=1.4)
petal_width  = st.number_input('Petal Width',  min_value=0.0, max_value=10.0, step=0.1, value=0.2)

if st.button('Prediksi'):
    # Masukkan urutan fitur sesuai domain model
    data = Orange.data.Table(
        domain, 
        [[sepal_length, sepal_width, petal_length, petal_width]]
    )
    
    # Prediksi: Orange classifier menggunakan call langsung
    pred = model(data)[0]
    label = domain.class_var.values[int(pred)]
    st.success(f"Hasil Prediksi: {label}")
