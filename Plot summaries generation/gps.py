# -----------IMAGE CAPTIONING-------------
import torch
from transformers import GitForCausalLM, AutoProcessor, AutoConfig

global img_root_path

img_root_path = 'C:\\Users\\caprattr\\repos\\AI-GENERATED-PLOT-SUMMARIES-FOR-CHILDRENS-STORIES\\Datasets\\childrens-books-test'

base_checkpoint = 'microsoft/git-large-r-coco'
processor = AutoProcessor.from_pretrained(base_checkpoint)

config = AutoConfig.from_pretrained(base_checkpoint)
model_greedy = GitForCausalLM(config)
model_beam = GitForCausalLM(config)

# Check if running on GPU or CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the pre-trained model
checkpoint_greedy = 'childrensimages-caption-20231108'
checkpoint_beam = 'childrensimages-caption-20231109'

if device == 'cuda':
    model_greedy.load_state_dict(torch.load(checkpoint_greedy))
    model_beam.load_state_dict(torch.load(checkpoint_beam))
elif device == 'cpu':
    model_greedy.load_state_dict(torch.load(checkpoint_greedy,map_location=torch.device('cpu')))
    model_beam.load_state_dict(torch.load(checkpoint_beam,map_location=torch.device('cpu')))
    
model_greedy.to(device)
model_beam.to(device)

print('Loaded Image Captioning models.')

def clean_sentence(sentence):
    return sentence.replace('drawing of ','').replace('picture of ','').\
        replace('image of ','').replace('\'','\\\'').replace('\\\'','\'')

from PIL import Image

# Function to generate caption
def generate_captions(img_path):
    print('Reading image...')
    image = Image.open(img_path)
    new_size = (224, 224)
    # Resize the image
    resized_image = image.resize(new_size)
    
    # prepare image for the model
    inputs = processor(images=resized_image, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values
    
    # Greedy search model
    print('Generating caption with greedy model...')
    generated_ids_greedy = model_greedy.generate(pixel_values=pixel_values, max_length=50)
    generated_caption_greedy = processor.batch_decode(generated_ids_greedy, skip_special_tokens=True)[0]
    generated_caption_greedy = clean_sentence(generated_caption_greedy)
    
    # Beam search model
    print('Generating caption with beam model...')
    generated_ids_beam = model_beam.generate(pixel_values=pixel_values, max_length=50)
    generated_caption_beam = processor.batch_decode(generated_ids_beam, skip_special_tokens=True)[0]
    generated_caption_beam = clean_sentence(generated_caption_beam)    
    
    return resized_image, [generated_caption_greedy, generated_caption_beam]

# -----------PLOT SUMMARY GENERATION-------------
import openai
# TODO: Hide key
openai.api_key = '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'

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
    print('Formatting caption {0} as an input command...'.format(idx))
    reply = call_chatgpt(instructions)
    if '{character 1}' in reply or '{character 2}' in reply or '{location}' in reply or '{theme}' in reply:
        reply = call_chatgpt(instructions)
    return reply

# Function to generate plot summary creation command
def generate_plot_input(base_instructions,captions=[]):
    print('Starting the plot input commands generation...')
    scratchplot_inputs = []
    for idx, caption in enumerate(captions):
        instructions = base_instructions + caption
        reply = format_caption(idx+1, instructions)
        scratchplot_inputs.append(reply)
    return scratchplot_inputs

# Function to generate plot summary
def generate_plot_summary(image_path, language='english'):
    
    if img_root_path != '':
        image_path = img_root_path + '\\' + image_path
    
    # Given image path, generate captions (greedy, beam)
    pil_image, captions = generate_captions(image_path)
    
    base_instructions = 'If existing, identify subjects, location and captions or title in the given sentence. Subjects will be called \"characters\" as we will generate a children\'s story based on the sentence content. If no subjects, location or captions or title are provided, based on the sentence generate them, remember to only generate the ones that are missing in given sentence; these fields cannot be blank and all subjects or characters, locations and main themes should be suitable for children\'s stories. With previous information obtained from given sentence, complete the following command by replacing the curly brackets with characters (1, 2, 3, etc.), location and main theme, modifying the command to match as many characters are identified: \"Write a plot summary of a children\'s story featuring {character 1} and {character 2} in {location} with the main title {title}.\" Example of a sentence: a boy and mom with a dog and the title \"Family love\". Generated command: \"Write a plot summary of a children\'s story featuring a boy and his mom and a dog in an enchanted mansion with the title \"Family love\". Please show only the generated command and follow the format exactly. Sentence to analyze is at the end of this command.'
           
    # Generate the plot input command
    plot_inputs = generate_plot_input(base_instructions, captions)
    print('Generated inputs:',plot_inputs)
    
    # For each plot input command generate a plot summary
    plot_security = ' As it is a story for children, it does not use discriminatory, offensive, racist, religious language, or any topic that incites violence or hatred.'
    plot_language = ' Write the output in ' + language
    plot_summaries = []
    for idx, pl_input in enumerate(plot_inputs):
        print('Generating plot summary for input command {0}...'.format(idx+1))
        plot_summaries.append(call_chatgpt(pl_input + plot_security + plot_language))
    
    print(plot_summaries)
    
    return {'pil_image': pil_image, 'captions': plot_summaries}