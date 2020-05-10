from HANN.han_classify import classify_documents
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def upload_file():
   return render_template('upload.html')

@app.route('/predict', methods= ['GET', 'POST'])
def predict():
    file = request.files['file']
    result = classify_documents(file)
    return result

if __name__ == '__main__':
    app.run()