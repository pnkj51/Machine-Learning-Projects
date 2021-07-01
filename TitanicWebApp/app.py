from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt 
import io, base64, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#defalut values
DEFAULT_EMBARKED = 'Southampton'
DEFAULT_FARE = 33
DEFAULT_AGE = 30
DEFAULT_GENDER = 'Female'
DEFAULT_TITLE = 'Mrs.'
DEFAULT_CLASS = 'Second'
DEFAULT_CABIN = 'C'
DEFAULT_SIBSP = 0
DEFAULT_PARCH = 0


regressor = LogisticRegression()

app = Flask(__name__)

# @app.before_first_request
# def modelbuilding():
# 	global average_survival_rate

# 	#model building
# 	#saving in pickle file

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/submit_new', methods=['POST'])
def submit_new():
	model_results = ''
	if request.method == 'POST':
		#get values form html form
		selected_embarked = request.form['selected_embarked']
		selected_fare = request.form['selected_fare']
		selected_age = request.form['selected_age']
		selected_gender = request.form['selected_gender']
		selected_title = request.form['selected_title']
		selected_class = request.form['selected_class']
		selected_cabin = request.form['selected_cabin']
		selected_sibsp = request.form['selected_sibsp']
		selected_parch = request.form['selected_parch']

		age = int(selected_age)
		isfemale = 1 if selected_gender == 'Female' else 0
		sibsp = int(selected_sibsp)
		parch = int(selected_parch)
		fare = int(selected_fare)


		embarked_Q = 1
		embarked_S = 0
		embarked_Unknown = 0
		embarked_nan = 0

		if selected_embarked[0] == 'Q':
			embarked_Q = 1
		if selected_embarked[0] == 'S':
			embarked_S = 1

		pclass_Second = 0
		pclass_Third = 0
		pclass_nan = 0
		if selected_class=='Second':
			pclass_Second = 0
		if selected_class=='Third':
			pclass_Third = 0

		title_Master = 0
		title_Miss = 0
		title_Mr = 0
		title_Mrs = 0
		title_Rev = 0
		title_Unknown = 0
		title_nan = 0
		if selected_title=='Master':
			title_Master=1
		if selected_title=='Miss.':
			title_Miss=1
		if selected_title=='Mr.':
			title_Mr=1
		if selected_title=='Mrs.':
			title_Mrs=1
		if selected_title=='Rev.':
			title_Rev=1
		if selected_title=='Unknown':
			title_Unknown=1

		cabin_B = 0
		cabin_C = 0
		cabin_D = 0
		cabin_E = 0
		cabin_F = 0
		cabin_G = 0
		cabin_T = 0
		cabin_Unknown = 0
		cabin_nan = 0

		if selected_cabin=='B':
			cabin_B = 1

		if selected_cabin=='C':
			cabin_C = 1
		
		if selected_cabin=='D':
			cabin_D = 1
		
		if selected_cabin=='E':
			cabin_E = 1
		
		if selected_cabin=='F':
			cabin_F = 1
		
		if selected_cabin=='G':
			cabin_G = 1
		
		if selected_cabin=='T':
			cabin_T = 1

		if selected_cabin=='Unknown':
			cabin_Unknown = 1
		

		#pass all values in list
		user_designed_passenger = [[age, sibsp, parch, fare, isfemale, pclass_Second, 
							pclass_Third, pclass_nan, cabin_B, cabin_T,cabin_G,cabin_F, 
							cabin_E, cabin_D,cabin_Unknown, cabin_nan,
							embarked_S, embarked_Q, embarked_nan, embarked_Unknown,
							title_Master, title_Unknown, title_Rev, title_Mrs, title_Mr,
							title_nan,title_Miss]]

		#predict probabilities
		ypred = regressor.predict_proba(user_designed_passenger)
		probability_of_surviving = ypred[0][1] * 100

		#plot the probabilities using matplotlib
		fig = plt.figure()
		objects = ('Average Survival Rate', 'Traveler')

		ypos = np.arange(len(objects))
		performance = [average_survival_rate, probability_of_surviving]

		ax = fig.add_subplot(111)
		colors = ['gray', 'blue']

		plt.bar(ypos, performance, align='center', color = colors, alpha=0.5)
		plt.xticks(ypos, objects)

		plt.axhline(average_survival_rate, color='r')
		plt.ylim([0,100])

		plt.ylabel('Survival Probability')
		plt.title('How did the traveler do? \n '+ str(round(probability_of_surviving,2)))

		img = io.BytesIO()
		plt.savefig(img, format='png')

		img.seek(0)

		plot_url = base64.b64encode(img.getvalue()).decode()


		#return all the values in the template
		return render_template('index.html',
			model_results= model_results,
			model_plot = Markup('<img src="data:image/png;base64,{}"> '.format(plot_url)),
			selected_embarked = selected_embarked,
			selected_fare = selected_fare,
			selected_age = selected_age,
			selected_gender = selected_gender,
			selected_title = selected_title,
			selected_class = selected_class,
			selected_cabin = selected_cabin,
			selected_sibsp = selected_sibsp,
			selected_parch = selected_parch)
	else:
		#return all the defalut values
		return render_template('index.html',
			model_results = '',
			model_plot = '',
			selected_embarked = DEFAULT_EMBARKED,
			selected_fare = DEFAULT_FARE,
			selected_age = DEFAULT_AGE,
			selected_gender = DEFAULT_GENDER,
			selected_title = DEFAULT_TITLE,
			selected_class = DEFAULT_CLASS,
			selected_cabin = DEFAULT_CABIN,
			selected_sibsp = DEFAULT_SIBSP,
			selected_parch = DEFAULT_PARCH
			)


if __name__ == '__main__':
	app.run(debug=True, port=5001)