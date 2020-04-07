from flask import Flask,render_template,request
app = Flask(__name__,template_folder='tamplates')
import pickle



with open('model.pkl','rb') as f:
    clf = pickle.load(f)

#@app.route('/')
#def vikas():
    #return render_template('front.html')

@app.route('/' , methods=["GET" ,"POST"])

def hello_world():
    
    
    if request.method == "POST":
        myDict = request.form
        Age = int(myDict['Age'])
        Fever = int(myDict['Fever'])
        BodyPains = int(myDict['BodyPains'])
        RunnyNose = int(myDict['RunnyNose'])
        Difficulty_in_Breath = int(myDict['Difficulty_in_Breath'])
        inputFeatures = [Age,Fever,BodyPains,RunnyNose,Difficulty_in_Breath]
        infProb =clf.predict_proba([inputFeatures])[0][1]
        print(infProb)
        if infProb > 0.49:
            report='positive' 
        else:
            report='negative'
        return render_template('show.html', inf=round(infProb *100),report=report) 
    return render_template('index.html')    
       
     
    #return 'Hello, World!' + str(infProb)
if __name__ == "__main__":
    app.run(debug=True)
