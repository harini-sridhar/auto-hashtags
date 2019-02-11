import sys
sys.path.append("./src")
import os
import uuid
import requests
from whitenoise import WhiteNoise

from flask import (Flask, flash, redirect, render_template, request,
                   send_from_directory, url_for)

from src import eval
import shutil


UPLOAD_FOLDER = './static/images/'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
SECRET_KEY = 'YOUR SECRET KEY FOR FLASK HERE'

app = Flask(__name__)
app.wsgi_app = WhiteNoise(app.wsgi_app, root='static/')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = SECRET_KEY

@app.route('/<path:path>')
def static_file(path):
    return app.send_static_file(path)

# check if file extension is right
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# force browser to hold no cache. Otherwise old result returns.
@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# main directory of programme
@app.route('/',methods=['GET', 'POST'])
def upload_file():
    try:
        # remove files created more than 5 minute ago
        #os.system("find /static/images/ -maxdepth 1 -mmin +5 -type f -delete")
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
        os.mkdir(UPLOAD_FOLDER)
    except OSError:
        pass
    if request.method == 'POST':
        # check if the post request has the file part
        if 'content-file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        content_file = request.files['content-file']
        files = [content_file]
        print(files)
        # give unique name to each image
        content_name = str(uuid.uuid4()) + ".png"
        file_names = [content_name]
        for i, file in enumerate(files):
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_names[i]))
        args={
            'image_folder' : "./static/images/",
            'model': 'src/model.pth',
            'infos_path': 'src/infos.pkl',
            'num_images': 1,
            'num_tags': 30
        }
        # returns created caption
        caption = eval.generate(args)
        #caption = "#pretty #beachhouse #beachbum #sky #blue #sunny #clouds #beachday #beachbody #cloudporn #beachlife #seaside #shore #view #nature #instabeach #sand #beach #beautiful #amazing #summer #fun #beachtime #instasummer #beaches #beachbound #group #people #kites #beach"

        params={
            'content': "./static/images/" + file_names[0],
            'caption': caption
        }
        return render_template('elements.html', **params)
    return render_template('upload.html')


@app.errorhandler(404)
def page_not_found(error):
    return render_template('page_not_found.html'), 404


if __name__ == "__main__":
    app.run(host='0.0.0.0')
