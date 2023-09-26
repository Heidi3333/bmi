import streamlit as st
if st.checkbox ("Yes/No"):
    st.text("show")
else:
    st.text("hide")

status =st.radio("Select your gender",("Male","female"))
st.write(status)


if  status =="Male":
    st.write("You  are a male")
else:
    st.write("You  are a female")

Selection =st.selectbox("Hobby",["Tennis","Football"])
List =st.multiselect("Hobby",["Tennis","Football"])

if st.button("please submit"):
    st.write(Selection)
    st.write(List)

Gender = st.text_input("please enter your gender")
st.write(Gender)
st.slider("Your age",18,30,step=2)

st.title('Welcome to BMI Calculator')
weight = st.slider("Your weight in kgs",20,100)
height = st.slider("Your height in cm",30,300)
if st.button("press to calculate your bmi"):
   bmi = weight / ((height / 100) ** 2)
   st.text(f"your bmi is {bmi}")
