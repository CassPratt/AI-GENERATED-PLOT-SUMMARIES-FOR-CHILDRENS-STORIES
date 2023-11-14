import logging

# Configure the logging module
logging.basicConfig(level=logging.INFO)
# -----------IMAGE CAPTIONING-------------
import torch
from transformers import GitForCausalLM, AutoProcessor, AutoConfig

global img_root_path

img_root_path = 'C:\\Users\\caprattr\\repos\\AI-GENERATED-PLOT-SUMMARIES-FOR-CHILDRENS-STORIES\\Datasets\\childrens-books-test'

# Load processor
base_checkpoint = 'microsoft/git-large-r-coco'
processor = AutoProcessor.from_pretrained(base_checkpoint)

# Load models from HuggingFace
checkpoint_greedy = 'CassPR/childrensimages-caption-20231108'
checkpoint_beam = 'CassPR/childrensimages-caption-20231109'

model_greedy = GitForCausalLM.from_pretrained(checkpoint_greedy)
model_beam = GitForCausalLM.from_pretrained(checkpoint_beam)

# Check if running on GPU or CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_greedy.to(device)
model_beam.to(device)

logging.info('Loaded Image Captioning models.')

def clean_sentence(sentence):
    return sentence.replace('drawing of ','').replace('picture of ','').\
        replace('image of ','').replace('\'','\\\'').replace('\\\'','\'')

from PIL import Image

# Function to generate caption
def generate_captions(pil_image):
    logging.info('Resizing image...')
    new_size = (224, 224)
    # Resize the image
    resized_image = pil_image.resize(new_size)
    
    # prepare image for the model
    inputs = processor(images=resized_image, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values
    
    # Greedy search model
    logging.info('Generating caption with greedy model...')
    generated_ids_greedy = model_greedy.generate(pixel_values=pixel_values, max_length=50)
    generated_caption_greedy = processor.batch_decode(generated_ids_greedy, skip_special_tokens=True)[0]
    generated_caption_greedy = clean_sentence(generated_caption_greedy)
    
    # Beam search model
    logging.info('Generating caption with beam model...')
    generated_ids_beam = model_beam.generate(pixel_values=pixel_values, max_length=50)
    generated_caption_beam = processor.batch_decode(generated_ids_beam, skip_special_tokens=True)[0]
    generated_caption_beam = clean_sentence(generated_caption_beam)    
    
    return [generated_caption_greedy, generated_caption_beam]

# -----------PLOT SUMMARY GENERATION-------------
import openai
import kv_secrets as kv

openai.api_key = kv.retrieve_secret('AI-GPSCS-OpenAIKey')

# Function to call ChatGPT with instructions
def call_chatgpt(instructions):
    # Set a context for the ChatGPT API
    messages = [ {"role": "system", "content": "You are an intelligent assistant."} ]
    if instructions:
        messages.append(
            {"role": "user", "content": instructions},
        )
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
      
    reply = chat.choices[0].message.content
    messages.append({"role": "assistant", "content": reply})
    
    return reply

# Function to format caption to generate a command
def format_caption(idx, instructions):    
    logging.info('Formatting caption {0} as an input command...'.format(idx))
    reply = call_chatgpt(instructions)
    if '{character 1}' in reply or '{character 2}' in reply or '{location}' in reply or '{theme}' in reply:
        reply = call_chatgpt(instructions)
    return reply

# Function to generate plot summary creation command
def generate_plot_input(base_instructions,captions=[]):
    logging.info('Starting the plot input commands generation...')
    scratchplot_inputs = []
    for idx, caption in enumerate(captions):
        instructions = base_instructions + caption
        reply = format_caption(idx+1, instructions)
        scratchplot_inputs.append(reply)
    return scratchplot_inputs

# Function to generate plot summary
def generate_plot_summary(pil_image, language='english'):
    
    # Given image path, generate captions (greedy, beam)
    captions = generate_captions(pil_image)
    
    base_instructions = 'If existing, identify subjects, location and captions or title in the given sentence. Subjects will be called \"characters\" as we will generate a children\'s story based on the sentence content. If no subjects, location or captions or title are provided, based on the sentence generate them, remember to only generate the ones that are missing in given sentence; these fields cannot be blank and all subjects or characters, locations and main themes should be suitable for children\'s stories. With previous information obtained from given sentence, complete the following command by replacing the curly brackets with characters (1, 2, 3, etc.), location and main theme, modifying the command to match as many characters are identified: \"Write a plot summary of a children\'s story featuring {character 1} and {character 2} in {location} with the main title {title}.\" Example of a sentence: a boy and mom with a dog and the title \"Family love\". Generated command: \"Write a plot summary of a children\'s story featuring a boy and his mom and a dog in an enchanted mansion with the title \"Family love\". Please show only the generated command and follow the format exactly. Sentence to analyze is at the end of this command.'
           
    # Generate the plot input command
    plot_inputs = generate_plot_input(base_instructions, captions)
    logging.info('Generated inputs:',plot_inputs)
    
    # For each plot input command generate a plot summary
    plot_security = ' As it is a story for children, it does not use discriminatory, offensive, racist, religious language, or any topic that incites violence or hatred.'
    plot_language = ' Write the output in ' + language
    plot_summaries = []
    for idx, pl_input in enumerate(plot_inputs):
        logging.info('Generating plot summary for input command {0}...'.format(idx+1))
        plot_summaries.append(call_chatgpt(pl_input + plot_security + plot_language))
    
    logging.info(plot_summaries)
    
    return plot_summaries