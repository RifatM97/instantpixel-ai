import os 
import streamlit as st
import yaml
import pandas as pd
import PIL.Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import time
import sys
import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    Image,
    Part,
)
import cv2
import joblib

# set initial starting dir
os.chdir(os.getcwd())

# Initialise vertexAI
PROJECT_ID = "vf-grp-eris-prd-main-01"  # @param {type:"string"}
LOCATION = "europe-west1"  # @param {type:"string"}
vertexai.init(project=PROJECT_ID, location=LOCATION)
multimodal_model = GenerativeModel("gemini-1.0-pro-vision")

# helper function for yaml
def modify_yaml(custom_prompt: str, phrase, box, output_folder, bg_image=None):
    # Load the data
    with open("prompt.yaml", 'r') as file:
        data = yaml.safe_load(file)

    # Modify a value
    data['prompt'] = custom_prompt  
    data['location'] = box  
    data['phrase'] = phrase  
    data['output_folder'] = output_folder
    if bg_image is not None:
        tmp_folder = 'generation_samples/temp_inpaint_im/'
        data['image'] = os.path.join(tmp_folder, bg_image.name)
    
    # Save the changes
    with open("prompt.yaml", 'w') as file:
        yaml.dump(data, file)

    print("YAML file modified successfully!")


col1, col2 = st.columns([0.2,0.8]) 
with col2:
    st.title("InstantPixel.AI Demo")
# with col1:
#     st.image("asset/final_logo.png", width=115)

# col1, col2 = st.columns([0.6,0.4]) 

# Side bar of tools
st.sidebar.title("Tools")
# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox("Drawing tool:", ("freedraw", "rect", "circle"))
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
realtime_update = st.sidebar.checkbox("Update in realtime", True)
phrase = st.sidebar.text_input("Key Words")

# save image for inpainting later
if bg_image:
    temp_fld = '/home/jupyter/InstantPixel/GLIGEN/generation_samples/temp_inpaint_im/'
    bg_image_path = os.path.join(temp_fld, bg_image.name)
    with open(bg_image_path, "wb") as f:
        f.write(bg_image.getvalue())

# with col1:
# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_color=stroke_color,
    stroke_width=3,
    background_image=PIL.Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=512,
    width=512,
    drawing_mode=drawing_mode,
    point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
    key="canvas",
)

# Do something interesting with the image data and paths
bboxes = []
if canvas_result.json_data is not None:
    objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str becaus PyArrow

    print(canvas_result.json_data["objects"]) 
    # Iterate through each object in the "objects" list
    for obj in canvas_result.json_data['objects']:
        # Extract the desired values
        x0 = int(obj['left'])/512
        y0 = int(obj['top'])/512
        x1 = (int(obj['left'])+int(obj['width']))/512
        y1 = (int(obj['top'])+int(obj['height']))/512

        # Create a list
        object_values = [x0, y0, x1, y1]

        # Append the list/tuple to the extracted_values list
        bboxes.append(object_values)

# Print the extracted values list
print(bboxes)

############# testing ##############

# # Creating a chatbot for simple prompts
# with st.sidebar:
#     messages = st.container(height=300)
#     if prompt := st.chat_input("You imagine, InstantPixel creates..."):
#         messages.chat_message("user").write(prompt)
#         messages.chat_message("assistant").write("Image Generation in Progress...")

#         new_path = "/home/jupyter/InstantPixel/GLIGEN/"
#         os.chdir(new_path)

#         # remove old images from the folder 
#         os.system("rm -r generation_samples/final_output/*")

#         # running GLIGEN
#         phrases = [phrase]
#         output_folder = "final_output"
#         modify_yaml(prompt, phrases, bboxes, output_folder)

#         # health check
#         print("Executing GLIGEN code...")

#         if bg_image:
#             with st.spinner('Image Generation in Progress...'):
#                 # os.system("python gligen_inference.py")
#                 st.info('This is a purely informational message', icon="ℹ️")
#             st.success('Generation Completed!')
#         else:
#             with st.spinner('Image Generation in Progress...'):
#                 os.system("python gligen_inference.py")
#             st.success('Generation Completed!')
#             messages.chat_message("assistant").write("Generation Completed!")

#        # health check
#         print("Generation Completed")

#         # uploading generated image
#         image_directory = 'generation_samples/final_output/'
#         image_path = os.path.join(image_directory, os.listdir(image_directory)[0])
#         st.image(image_path, caption='Generated Image')

# with col2:
# creating a Gemini pro bot
new_chat_id = f'{time.time()}'
MODEL_ROLE = 'InstaPixel.AI'
AI_AVATAR_ICON = '🤖'
# Create a data/ folder if it doesn't already exist
try:
    new_path = "/home/jupyter/InstantPixel/"
    os.chdir(new_path)
    os.mkdir('data/')
except:
    pass

# Load past chats (if available)
try:
    past_chats: dict = joblib.load('data/past_chats_list')
except:
    past_chats = {}

# Sidebar allows a list of past chats
with st.sidebar:
    new_size = (70, 70)
    # Load the image and resize it
    image = PIL.Image.open('asset/final_logo.png')
    image = image.resize(new_size)
    # image = cv2.imread('asset/final_logo.png')
    # image = cv2.resize(image, new_size)

    col1, col2 = st.columns(2)

    st.write('# Past Chats')
    if st.session_state.get('chat_id') is None:
        st.session_state.chat_id = st.selectbox(
            label='Pick a past chat',
            options=[new_chat_id] + list(past_chats.keys()),
            format_func=lambda x: past_chats.get(x, 'New Chat'),
            placeholder='_',
        )
    else:
        # This will happen the first time AI response comes in
        st.session_state.chat_id = st.selectbox(
            label='Pick a past chat',
            options=[new_chat_id, st.session_state.chat_id] + list(past_chats.keys()),
            index=1,
            format_func=lambda x: past_chats.get(x, 'New Chat' if x != st.session_state.chat_id else st.session_state.chat_title),
            placeholder='_',
        )
    # Save new chats after a message has been sent to AI
    # TODO: Give user a chance to name chat
    st.session_state.chat_title = f'ChatSession-{st.session_state.chat_id}'

full_response = ''
with st.expander('# Chat with InstantPixel.AI'):
    st.write('## InstantPixel.AI')

    # Chat history (allows to ask multiple questions)
    try:
        st.session_state.messages = joblib.load(
            f'data/{st.session_state.chat_id}-st_messages'
        )
        st.session_state.gemini_history = joblib.load(
            f'data/{st.session_state.chat_id}-gemini_messages'
        )
        print('old cache')
    except:
        st.session_state.messages = []
        st.session_state.gemini_history = []
        print('new_cache made')

    st.session_state.model = GenerativeModel("gemini-1.0-pro-vision")
    st.session_state.chat = st.session_state.model.start_chat(
        history=st.session_state.gemini_history,
    )

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(
            name=message['role'],
            avatar=message.get('avatar'),
        ):
            st.markdown(message['content'])

    # React to user input
    if prompt := st.chat_input('Your message here...'):
        scenario = prompt
        # Save this as a chat for later
        if st.session_state.chat_id not in past_chats.keys():
            past_chats[st.session_state.chat_id] = st.session_state.chat_title
            joblib.dump(past_chats, 'data/past_chats_list')
        # Display user message in chat message container
        with st.chat_message('user'):
            st.markdown(scenario)
        # Add user message to chat history
        st.session_state.messages.append(
            dict(
                role='user',
                content=scenario,
            )
        )

        prompt_template = """
        You are a friently creative design assistant helping users create image generation prompts for marketing campaigns.
        When they user provides a scenario, you should provide a suggestion of a short and concise image generation 
        prompt based on the given scenario.
        Confirm with the user if they want to proceed with the prompt you provided. If they say yes, 
        you must only respond with just the prompt again.

        """
        # scenario = "**[A fish laying on the desert]**"  # Replace with user input mechanism
        prompt_scene = prompt_template + scenario

        # Send message to AI
        response = st.session_state.chat.send_message(
            prompt_scene,
            stream=True,
        )

        # Display assistant response in chat message container
        with st.chat_message(
            name=MODEL_ROLE,
            avatar=AI_AVATAR_ICON,
        ):
            message_placeholder = st.empty()
            # full_response = ''
            assistant_response = response
            # Streams in a chunk at a time
            for chunk in response:
                # Simulate stream of chunk
                # TODO: Chunk missing `text` if API stops mid-stream ("safety"?)
                for ch in chunk.text.split(' '):
                    full_response += ch + ' '
                    time.sleep(0.05)
                    # Rewrites with a cursor at end
                    message_placeholder.write(full_response + '▌')
            # Write full message with placeholder
            message_placeholder.write(full_response)
         
        # running GLIGEN if pressed on button
        print("final_prompt:", full_response)
        if bg_image is None:
            if st.button('Generate Image', on_click=lambda: generate_image_gligen(full_response, phrase, bboxes)):
                pass
            show_output_image()

        elif bg_image is not None:
            if st.button('Generate Image', on_click=lambda: inpaint_image_gligen(full_response, phrase, bboxes, bg_image)):
                pass
            show_output_image()

        # uploading generated image
        
        
#### testing phase
#            if full_response == "Thanks for using InstantPixel.AI!":
#                print("is this running? ", full_response)
#                # running Gligen if button is pressed
#                st.button('Generate Image')
#                         new_path = "/home/jupyter/InstantPixel/GLIGEN/"
#                         os.chdir(new_path)

#                         # remove old images from the folder 
#                         os.system("rm -r generation_samples/final_output/*")

#                         # running GLIGEN
#                         phrases = [phrase]
#                         output_folder = "final_output"
#                         modify_yaml(full_response, phrases, bboxes, output_folder)

#                         # health check
#                         print("Executing GLIGEN code...")

#                         if bg_image:
#                             with st.spinner('Image Generation in Progress...'):
#                                 # os.system("python gligen_inference.py")
#                                 st.info('This is a purely informational message', icon="ℹ️")
#                             # st.success('Generation Completed!')
#                         else:
#                             with st.spinner('Image Generation in Progress...'):
#                                 os.system("python gligen_inference.py")
#                             st.success('Generation Completed!')
#                             # messages.chat_message("assistant").write("Generation Completed!")

#                        # health check
#                         print("Generation Completed")

#                         # uploading generated image
#                         image_directory = 'generation_samples/final_output/'
#                         image_path = os.path.join(image_directory, os.listdir(image_directory)[0])
#                         st.image(image_path, caption='Generated Image')


        # Add assistant response to chat history
        st.session_state.messages.append(
            dict(
                role=MODEL_ROLE,
                content=st.session_state.chat.history[-1].parts[0].text,
                avatar=AI_AVATAR_ICON,
            )
        )
        st.session_state.gemini_history = st.session_state.chat.history


        # Save to file
        joblib.dump(
            st.session_state.messages,
            f'data/{st.session_state.chat_id}-st_messages',
        )
        joblib.dump(
            st.session_state.gemini_history,
            f'data/{st.session_state.chat_id}-gemini_messages',
        )


def generate_image_gligen(full_response, phrase, bboxes):
    
    new_path = "/home/jupyter/InstantPixel/GLIGEN/"
    os.chdir(new_path)

    # remove old images from the folders
    os.system("rm -r generation_samples/final_output/*")

    # health check
    print("Executing GLIGEN code...")

    print("test2333333333_generate :",full_response)
    # running generation model
    phrases = [phrase]
    output_folder = "final_output"
    modify_yaml(full_response, phrases, bboxes, output_folder, bg_image=None)
    with st.spinner('Image Generation in Progress...'):
        os.system("python gligen_inference.py")
    st.success('Generation Completed!')

    # health check
    print("Generation Completed")
    
def inpaint_image_gligen(full_response, phrase, bboxes, bg_image):
    
    new_path = "/home/jupyter/InstantPixel/GLIGEN/"
    os.chdir(new_path)
    
    # remove old images from the folders
    os.system("rm -r generation_samples/final_output/*")
    
    # health check
    print("Executing GLIGEN code...")
    
    # running inpainting model
    phrases = [phrase]
    output_folder = "final_output"
    modify_yaml(full_response, phrases, bboxes, output_folder, bg_image)
    with st.spinner('Image Generation in Progress...'):
        os.system("python gligen_inpaint_inference.py")
    st.success('Generation Completed!')
    
    # health check
    print("Generation Completed")
    
    # empty temp bg image folder
    os.system("rm -r generation_samples/temp_inpaint_im/*")
    

# # running GLIGEN if pressed on button
# print("final_prompt:", full_response)
# if bg_image is None:
#     if st.button('Generate Image', on_click=lambda: generate_image_gligen(full_response, phrase, bboxes)):
#         pass

# elif bg_image is not None:
#     if st.button('Generate Image', on_click=lambda: inpaint_image_gligen(full_response, phrase, bboxes, bg_image)):
#         pass

# uploading generated image
def show_output_image():
    new_path = "/home/jupyter/InstantPixel/GLIGEN/"
    os.chdir(new_path)
    image_directory = 'generation_samples/final_output/'
    image_path = os.path.join(image_directory, os.listdir(image_directory)[0])
    st.image(image_path, caption='Generated Image')