"""
# My first app
Here's out first attempt at using data to create a table"
"""

import streamlit as st
import pandas as pd
import numpy as np

st.write(pd.DataFrame({
    'first column': [1,2,3,4],
    'second column':[10,20,30,40]
}))

dataframe = np.random.randn(10,20)
st.dataframe(dataframe)

dataframe = pd.DataFrame(
    np.random.randn(10,5),
    columns=('col %d' % i for i in range(5)))

st.write("static table example");
st.table(dataframe)

st.write("as dataframe")
st.dataframe(dataframe.style.highlight_max(axis=0))

st.write(" First attempt at using data to create a table:")

st.text(" This is st.text changed one more" )
#st.line_chart(" this is line chart ")

#Draw a line chart
chart_data = pd.DataFrame(
    np.random.randn(20,3),
    columns=['a','b','c'])
st.line_chart(chart_data)

#Plot a map
map_data = pd.DataFrame(
    np.random.randn(1000,2)/[50,50] + [37.76, -122.4],
    columns=['lat','lon'])
st.map(map_data)



