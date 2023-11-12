# -----------BASE CODE FOR GENERATION------------
import gps

# -----------HANDLING UI-------------

from flask import Flask, render_template, request
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    if 'selected_file' in request.files:
        selected_file = request.files['selected_file']
        filename = selected_file.filename

        # Call gps module generate_plot_summary function
        result = gps.generate_plot_summary(filename)
        #result = {'pil_image': selected_file,'captions':['TEXT 1','TEXT 2']}
        return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
