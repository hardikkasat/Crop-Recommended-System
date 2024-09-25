from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the pre-trained Random Forest model
RF_model = pickle.load(open('RF_model.pkl', 'rb'))

def classify(answer):
    return answer[0] + " is the ideal crop for cultivation in this area."

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        # Retrieve form data
        sn = float(request.form['sn'])
        sp = float(request.form['sp'])
        pk = float(request.form['pk'])
        pt = float(request.form['pt'])
        phu = float(request.form['phu'])
        pPh = float(request.form['pPh'])
        pr = float(request.form['pr'])
        
        # Prepare inputs and make prediction
        inputs = [[sn, sp, pk, pt, phu, pPh, pr]]
        prediction = RF_model.predict(inputs)
        result = classify(prediction)
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
