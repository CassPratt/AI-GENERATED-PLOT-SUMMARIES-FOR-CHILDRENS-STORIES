from flask import Flask, render_template, request
from PIL import Image
from io import BytesIO
import base64
import ai_gpscs as gps

app = Flask(__name__)

# Set a maximum file size (16 MB in this example)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def process_image(selected_file):
    try:
        original_image = Image.open(selected_file)
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

    # Resize the image to 100x100 pixels
    resized_image = original_image.resize((100, 100))

    # Convert the resized image to a base64 string
    buffered = BytesIO()
    resized_image.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return encoded_image, selected_file.filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    if 'selected_file' not in request.files:
        return render_template('error.html', error='No file part')

    selected_file = request.files['selected_file']

    try:
        encoded_image, filename = process_image(selected_file)
    except ValueError as e:
        return render_template('error.html', error=str(e))
    
    # Get the selected language from the form
    language = request.form.get('language', 'english')  # Default to English if not provided

    # Additional strings in the result
    #additional_strings = ['String 1', 'String 2']
    additional_strings = gps.generate_plot_summary(filename,language)

    result = {
        'filename': filename,
        'encoded_image': encoded_image,
        'additional_strings': additional_strings,
    }

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
