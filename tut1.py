import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.markdown(
    """
    <style>
     .main {
     background-color: #F5F5F5;

     }

    </style>
    """,
    unsafe_allow_html=True
)


@st.cache
def get_data():
    taxi_data = pd.read_csv('taxi_data.csv')
    return taxi_data



header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
modaltraining = st.beta_container()


with header:
    st.title("Welcome!")
    st.text("in this proj, i look into...")

with dataset:
    st.header("my dataset")
    st.text("i found this dataset at ...")

    taxi_data = get_data()
    st.write(taxi_data.head())

    st.subheader("Pick Up Location distribution")
    station_dist = pd.DataFrame( taxi_data['PULocationID'].head(50).value_counts() )
    st.bar_chart(station_dist)


with features:
    st.header("features i created")
    st.markdown("* ***first feature*** i created this feature because of this... ")
    st.markdown("* ***second feature*** i created this feature because of this... ")

with modaltraining:
    st.header("time to train modal")
    st.text("choose hyperparameters .....")


    sel_col, display_col = st.beta_columns(2)

    max_depth = sel_col.slider('what should be the max depth of model', min_value=10, max_value=100, value=20, step=10)
    n_estimators = sel_col.selectbox('how many trees should there be?', options=[100,200,300, 'No limit'], index=0 )

    sel_col.text('List of features in data')
    sel_col.write(taxi_data.columns)
    # st.table(taxi_data.columns)
    input_feature = sel_col.text_input("Which feature should be used as input feature?", "PULocationID" )

    if n_estimators=='No limit':
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    X = taxi_data[[input_feature]]
    y = taxi_data[['trip_distance']]

    regr.fit(X, y) 
    prediction = regr.predict(y) 
    display_col.subheader('Mean absolute error:') 
    display_col.write(mean_absolute_error(y, prediction)) 

    display_col.subheader('Mean squared error:') 
    display_col.write(mean_squared_error(y, prediction)) 

# import base64

# @st.cache(allow_output_mutation=True)
# def get_base64_of_bin_file(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# def set_png_as_page_bg(png_file):
#     bin_str = get_base64_of_bin_file(png_file)
#     page_bg_img = '''
#     <style>
#     body {
#     background-image: url("data:image/png;base64,%s");
#     background-size: cover;
#     }
#     </style>
#     ''' % bin_str
    
#     st.markdown(page_bg_img, unsafe_allow_html=True)
#     return

# set_png_as_page_bg('background.png')