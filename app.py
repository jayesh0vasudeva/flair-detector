from flask import flash, Flask, render_template, url_for, request, redirect
import praw
import re
from fastai2.text.all import *
import pandas
from werkzeug.utils import secure_filename
import os

client_id = 'F9GtKcaO_sk9xw'
secret = 'hRtKBcb40jnXhQGUZFX8OHCpkxY'

reddit = praw.Reddit(user_agent='funky_cool_bruh', client_id=client_id, client_secret=secret,
                     username='jayesh0vasudeva', password='jaimatadi88JAI')

def fetch_from_url(url):
	fetched = reddit.submission(url=url)
	return fetched.title


# path = Path('.')
df = pd.read_csv('titles_only.csv')
df = df.drop(['Unnamed: 0'], axis=1)
# for flair in 


# In[7]:


n = df.count()[0]
is_val = [True for i in range(df.count()[0])]
idx = [random.randint(0, n-1) for i in range(int(0.8*n))]


# In[8]:


for i in idx:
	is_val[i] = False

# len(is_val) == n


# In[9]:


df['is_valid'] = is_val


# In[11]:


df = df.rename(columns={"title": "text", "flair": "label"})
# df.head()


# In[12]:

print(df.label.count())


db = DataBlock(blocks=TextBlock.from_df('text', is_lm=True),
get_x=ColReader('text')).dataloaders(df)#, bs=64, seq_len=72)


# In[13]:


# dls = db.dataloaders(df, bs=64, seq_len=72)
# dls.show_batch(max_n=6)


# In[14]:


dls_clas = DataBlock(blocks=(TextBlock.from_df('text',seq_len=72, vocab=db.vocab), CategoryBlock),
get_x=ColReader('text'),
get_y=ColReader('label')).dataloaders(df)


# In[15]:


# dls_clas = db_clas.dataloaders(df, bs=64)
# dls_clas.show_batch(max_n=6)


# In[42]:


learn = text_classifier_learner(dls_clas, AWD_LSTM)


# In[43]:


# learn.fit_one_cycle(1, 2e-2)


# In[44]:


# learn.save('1epoch')


# In[45]:


# learn = learn.load('1epoch')


# In[46]:


# learn.unfreeze()
# learn.fit_one_cycle(10, 2e-3)


# In[47]:


# learn.unfreeze()
# learn.fit_one_cycle(6, 2e-3)


# In[48]:


# learn.save_encoder('titles_model')


# In[49]:


# learn.lr_find()


# In[81]:


# from fastai2.vision.all import *
# preds,y,losses = learn.get_preds(with_loss=True)
# interp = ClassificationInterpretation(dls_clas, preds, y , losses)


# In[50]:


learn = learn.load_encoder('titles_model')
learn.fit_one_cycle(1, 2e-3)

learn.save_encoder('titles_model')

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():

	if request.method == 'POST':

		url = request.form.get("email_name")
		data = fetch_from_url(url=url)
		print(data)
		pred = learn.predict(data)
		# print(pred)
		return render_template('predict.html', predictions=pred[0])

UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/uploads/'
DOWNLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/downloads/'

@app.route('/automated_testing',methods=['GET', "POST"])
def automated_testing():

	if request.method == 'POST':

	   if 'file' not in request.files:
	       print('No file attached in request')
	       return redirect(request.url)
	   file = request.files['file']
	   if file.filename == '':
	       print('No file selected')
	       return redirect(request.url)
	   if file and allowed_file(file.filename):
	       filename = secure_filename(file.filename)
	       file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
	       process_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), filename)

	       return redirect(url_for('uploaded_file', filename=filename))

	return render_template('automated_testing.html')

	# if request.method == 'POST':

	# 	if request.files:

	# 		txt = request.files["file"]

	# 		if txt.filename == "":

	# 			print("T")

	# 			return redirect(request.url)

	# 		txt.save(os.path.join(app.config["DOCUMENT_UPLOADS"], txt.filename))

	# 	result = request.files["file"]

	# else:

	# 	reult = request.args.get["file"]

		# return render_template('automated_testing.html')

	# else:
	# 	result = request.args.get('file')

	# return result



if __name__ == '__main__':
	app.run(debug=True)


# def maku_is_pagal(pgl, mental):
# 	return pgl+mental

# dumbness = maku_is_pagal(10, 100)
# print(dumbness)






