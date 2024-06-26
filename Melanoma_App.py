import streamlit as st
import numpy as np
from model_run import CNN
from PIL import Image

model = CNN()

general = '''**Protect your skin from the sun:**\n This is the most important thing you can do to reduce your risk of melanoma. Be sure to wear sunscreen with an SPF of 30 or higher every day, even on cloudy days.
    \n\n**Avoid tanning beds:**\n Tanning beds emit ultraviolet (UV) rays, which can damage your skin and increase your risk of melanoma.
    \n\n**Get regular skin exams:**\n A dermatologist can check your skin for signs of melanoma.\n
    \n**Cover up:**\n Wear protective clothing, such as hats and sunglasses, when you're outdoors.
    \n\n**Seek Professional help:**\n Do a monthly skin self-exam to check for any new or changing moles.
    \n\n**Resist the urge to scrub your skin clean:**\n If you've been sweating heavily or have a serious acne flare, it may seem natural to scrub your skin. Don't! Scrubbing irritates your skin, which can worsen any skin condition, including acne.
    \n\n**Stress less:**\n Finding healthy ways to manage stress can help your skin, too. Some skin diseases like psoriasis and atopic dermatitis (eczema) often appear for the first time when someone feels really stressed. Stress can also cause flare-ups of many skin conditions, including acne, eczema, psoriasis, and rosacea.
    \n\n**Don't smoke:**\n Smoking speeds up how quickly your skin ages. If you smoke, your wounds will also take longer to heal. And research shows that smoking worsens some skin diseases, including psoriasis and hidradenitis suppurativa.
    '''

warning = '''It is important to emphasize that this is a warning and not a confirmed diagnosis. \n
Advise the user to see a doctor as soon as possible.\n
Stress that the model is not a replacement for professional medical advice.\n'''

st.title("Skin Melanoma Assistant")
img = st.file_uploader("Please provide a clear image of the Skin patch", type="jpeg")

if img is not None:
    display_image = img.read()
    image = Image.open(img)
    img = np.array(image)
    image.close()

    prediction, message, note = model.predict(img)

    col1, col2 = st.columns(2)
    # col1.write("")
    # col1.write("It is recommended to try the")
    col1.caption("Make sure the image is clear")
    col1.image(display_image, use_column_width="auto", caption="Uploaded Image")
    col2.subheader(prediction)
    col2.write(message)

    if prediction == "Malignant Melanoma":
        st.divider()
        st.subheader("Keep in Mind :")
    st.write(note)

st.divider()
st.subheader("General Advisory for skin protection")
st.markdown(general)

if img is not None:
    st.divider()
    st.caption("This model is for informational purposes only and should not be used as a substitute for professional medical advice.")
    st.caption(warning)