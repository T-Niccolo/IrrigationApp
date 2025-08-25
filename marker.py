import streamlit as st

# Initialize session state once
if "value" not in st.session_state:
    st.session_state.value = 5

# Callback: when number input changes
def on_number_change():
    st.session_state.value = st.session_state.num_input

# Callback: when slider changes
def on_slider_change():
    st.session_state.value = st.session_state.slider

# Number input with callback
st.number_input(
    "Number Input",
    min_value=0,
    max_value=100,
    value=st.session_state.value,
    key="num_input",
    on_change=on_number_change
)

# Slider with callback
st.slider(
    "Slider",
    min_value=0,
    max_value=100,
    value=st.session_state.value,
    key="slider",
    on_change=on_slider_change
)

# Debug output
st.write("🔢 Shared value:", st.session_state.value)
