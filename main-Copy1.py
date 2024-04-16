import os 
import streamlit as st
import yaml
import pandas as pd
from PIL import Image as PILImage
from streamlit_drawable_canvas import st_canvas
import time
import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    Image,  # Assuming this is needed for other parts not shown
    Part,
)
import cv2
import joblib

# Initialize Vertex AI
vertexai.init(project="vf-grp-eris-prd-main-01", location="europe-west1")
multimodal_model = GenerativeModel("gemini-1.0-pro-vision")

def modify_yaml(custom_prompt: str, phrase, box, output_folder):
    with open("prompt.yaml", 'r') as file:
        data = yaml.safe_load(file)
    data['prompt'] = custom_prompt  
    data['location'] = box  
    data['phrase'] = phrase  
    data['output_folder'] = output_folder
    with open("prompt.yaml", 'w') as file:
        yaml.dump(data, file)
    print("YAML file modified successfully!")

drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
)
point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3) if drawing_mode == 'point' else 0
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
realtime_update = st.sidebar.checkbox("Update in realtime", True)
phrase = st.sidebar.text_input("Key Words")

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_color=stroke_color,
    stroke_width=3,
    background_image=PILImage.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=512,
    width=512,
    drawing_mode=drawing_mode,
    point_display_radius=point_display_radius,
    key="canvas",
)

bboxes = []
if canvas_result.json_data is not None:
    objects = pd.json_normalize(canvas_result.json_data["objects"])
    for obj in canvas_result.json_data['objects']:
        x0 = int(obj['left'])/512
        y0 = int(obj['top'])/512
        x1 = (int(obj['left'])+int(obj['width']))/512
        y1 = (int(obj['top'])+int(obj['height']))/512
        bboxes.append([x0, y0, x1, y1])

print(bboxes)

new_chat_id = f'{time.time()}'
MODEL_ROLE = 'InstaPixel.AI'
AI_AVATAR_ICON = '🤖'

try:
    new_path = "/home/jupyter/InstantPixel/"
    os.chdir(new_path)
    os.makedirs('data/', exist_ok=True)
except Exception as e:
    print(f"Error creating data directory: {e}")

try:
    past_chats: dict = joblib.load('data/past_chats_list')
except Exception as e:
    past_chats = {}
    print(f"Error loading past chats: {e}")

with st.sidebar:
    new_size = (70, 70)
    image = cv2.imread('asset/logo.png')
    col1, col2 = st.columns(2)
    st.write('# Past Chats')
    chat_options = [new_chat_id] + list(past_chats.keys())
    st.session_state.chat_id = st.selectbox(
        label='Pick a past chat',
        options=chat_options,
        format_func=lambda x: past_chats.get(x, 'New Chat'),
        index=chat_options.index(st.session_state.get('chat_id', new_chat_id)),
    )
    st.session_state.chat_title = f'ChatSession-{st.session_state.chat_id}'

with st.expander('# Chat with InstantPixel.AI'):
    st.write('## InstantPixel.AI')
    try:
        st.session_state.messages = joblib.load(f'data/{st.session_state.chat_id}-st_messages')
        st.session_state.gemini_history = joblib.load(f'data/{st.session_state.chat_id}-gemini_messages')
        print('Loaded old cache')
    except Exception as e:
        print('New cache made, reason:', e)

    st.session_state.chat = st.session_state.model.start_chat(history=st.session_state.gemini_history)
    for message in st.session_state.messages:
        with st.chat_message(name=message['role'], avatar=message.get('avatar')):
            st.markdown(message['content'])

    if prompt := st.chat_input('Your message here...'):
        if st.session_state.chat_id not in past_chats.keys():
            past_chats[st.session_state.chat_id] = f'ChatSession-{st.session_state.chat_id}'
            joblib.dump(past_chats, 'data/past_chats_list')
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)

        response = st.session_state.chat.send_message(prompt, stream=True)
        with st.chat_message(name=MODEL_ROLE, avatar=AI_AVATAR_ICON):
            full_response = ''
            for chunk in response:
                full_response += chunk.text + ' '
                st.markdown(full_response + '▌')
            st.session_state.messages.append({'role': MODEL_ROLE, 'content': full_response})

        joblib.dump(st.session_state.messages, f'data/{st.session_state.chat_id}-st_messages')
        st.session_state.gemini_history = st.session_state.chat.history
        joblib.dump(st.session_state.gemini_history, f'data/{st.session_state.chat_id}-gemini_messages')

    if st.button('Generate Image'):
        new_path = "/home/jupyter/InstantPixel/GLIGEN/"
        os.chdir(new_path)
        os.system("rm -r generation_samples/final_output/*")
        phrases = [phrase]
        output_folder = "final_output"
        modify_yaml(full_response_text, phrases, bboxes, output_folder)
        print("Executing GLIGEN code...")
        with st.spinner('Image Generation in Progress...'):
            os.system("python gligen_inference.py")
        st.success('Generation Completed!')
        image_directory = 'generation_samples/final_output/'
        image_path = os.path.join(image_directory, os.listdir(image_directory)[0])
        st.image(image_path, caption='Generated Image')
    else:
        st.write('Goodbye')
