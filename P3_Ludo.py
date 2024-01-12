from flask import Flask, flash, render_template, request
from database import get_db
import numpy as np
import pandas as pd
import requests, os, re, base64, pickle
from werkzeug.utils import secure_filename
from keras.datasets import mnist
from sklearn.linear_model import LogisticRegression
from PIL import Image
from io import BytesIO
from json import dumps
import cv2

UPLOAD_FOLDER = 'C:\\Users\\Simplon\\Documents\\Simplon\\Flask_project\\files_upload'
ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}

MyApp = Flask(__name__)
MyApp.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

#Page d'accueil : affichage simple, avec les liens pour naviguer sur les autres pages
@MyApp.route("/")
def accueil():
    return render_template("accueil.html")

#Page de gestion des utilisateurs, avec possibilité de créer de nouveaux utilisateurs en BDD
@MyApp.route("/users", methods=['GET', 'POST'])
def users():
    if request.method == 'POST':
        #Création d'un nouvel utilisateur après avoir saisi les informaitons dans le formulaire
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        gender = request.form['gender']
        pseudo = request.form['pseudo']

        db = get_db()
        curseur = db.cursor()

        #Préalable : on vérifie que le pseudo saisi n'est pas déjà utilisé
        query = f"select id from users where pseudo = '{pseudo}'"
        nb_res = False
        curseur.execute(query)
        print(query)
        for row in curseur:
            nb_res = True
            break
        if nb_res == True:
            #Si pseudo déjà utilisé : on affiche un message, et on remet le formulaire de saisie
            flash("Ce pseudo est déjà utilisé", "error")
            curseur.close()
            db.close()
            return render_template("users.html")
        else:
            #Si pseudo dispo, on enregistre l'utilisateur dans la base
            curseur.reset()
            query = f"insert into users (nom, prenom, pseudo, gender) values ('{lastname}', '{firstname}', '{pseudo}', '{gender}')"
            curseur.execute(query)
            db.commit()
            curseur.close()
            db.close()
            #Et on affiche un message + le formulaire de saisie
            if gender == "male":
                phrase = f"Utilisateur {firstname} {lastname} correctement enregistré avec le pseudo {pseudo}"
            else:
                phrase = f"Utilisatrice {firstname} {lastname} correctement enregistrée avec le pseudo {pseudo}"
            flash(phrase, "success")
            return render_template("users.html")
    #Si on est en GET, c'est l'affichage initial du formulaire
    return render_template("users.html")

#Page d'affichage de la liste des utilisateurs
@MyApp.route("/list_users")
def list_users():
    db = get_db()
    curseur = db.cursor()
    query = "select nom,prenom,gender,pseudo from users"
    curseur.execute(query)
    users = curseur.fetchall()
    curseur.close()
    db.close()
    return render_template("list_users.html", object=users)

#Page permettant de retrouver les dernières actualités d'une entreprise
@MyApp.route("/list_news", methods=['GET', 'POST'])
def list_news():
    if request.method == 'POST':
        action = request.form['action']
        if action == "search":
            #L'utilisateur a saisi le nom d'une entreprise dans le premier champs
            company = request.form['company']
            #On lance une prmeière recherche via l'API pour récupérer la liste des 'ticker' correspondant
            url = 'https://devapi.ai/api/v1/markets/search'
            params = { 'search': company }
            headers = { 'Authorization': 'Bearer 344|O4Z4GEwjtJGnhf9hXCphD7xgER6ZtokvpToWq0pq' }
            response = requests.request('GET', url, headers=headers, params=params)
            entities = response.json()["body"]
            if entities == []:
                #La recherche n'a rien trouvé, on réaffiche le formulaire de recherche
                flash("Votre recherche n'a rien donné")
                return render_template("list_news.html")
            else:
                #On crée et affiche un deuxième formulaire à l'utilisateur,
                #lui permettant de sélectionner l'entreprise qui l'intéresse parmi celles trouvées
                html = "<form action='/list_news' method='POST'>"
                html += "<input type='hidden' name='action' value='news' />"
                html += "<label for='entities'>Sélectionnez une entreprise :</label><select name='entites' id='entities'>"
                for entity in entities:
                    html += "<option value='" + entity["symbol"] + "'>" + entity["name"] + "</option>"
                html += "</select><input type='submit' value='Rechercher' /></form>"
                return render_template("list_news.html", reponse_search = html)
        elif action == "news":
            #L'utilisateur a choisi une entreprise dans le deuxième formulaire
            ticker_entite = request.form['entites']
            #On lance la recherche des actus via l'API
            url = 'https://devapi.ai/api/v1/markets/news'
            params = { 'ticker': ticker_entite }
            headers = { 'Authorization': 'Bearer 344|O4Z4GEwjtJGnhf9hXCphD7xgER6ZtokvpToWq0pq' }
            response = requests.request('GET', url, headers=headers, params=params)
            news = response.json()["body"]
            #On enregistre dans la BDD cette recherche
            db = get_db()
            curseur = db.cursor()
            query = f"insert into search_logs (user_pseudo, search_date, ticker, company, nb_news) values ('anonymous', now(), '{ticker_entite}', 'unknown', '{len(news)}')"
            curseur.execute(query)
            db.commit()
            curseur.close()
            db.close
            #On affiche les résultats à l'utilisateur
            html = ""
            for info in news:
                html += "<div><h2>" + info["title"] + "</h2><p>" + info["description"] + "</p>"
                html += "<p style='float: right;'>" + info["pubDate"] + "</p><a href='" + info["link"] + "' target='_blank'>Source</a></div>"
            return render_template("list_news.html", reponse_news = html)
    return render_template("list_news.html")

#Page pour l'nalyse de fichiers CSV et Excel
MyApp.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@MyApp.route("/files_stat",methods=["GET", "POST"])
def files_stat():
    if request.method == 'POST':
        #On vérifie que l'utilisateur a bien sélectionner un fichier
        if 'fichier' not in request.files:
            return render_template("files_stat.html", message = "Veuillez choisir un fichier !")
        fichier = request.files['fichier']
        if fichier and allowed_file(fichier.filename):
            filename = os.path.join(MyApp.config['UPLOAD_FOLDER'], secure_filename(fichier.filename))
            fichier.save(filename)
            vardump, extension = os.path.splitext(filename)
            print(filename)
            print(extension)
            if extension == ".csv":
                df = pd.read_csv(filename)
                return render_template("files_stat.html", tables=[df.describe().T.to_html(classes='data', index = True, justify = 'left')])
            elif extension == ".xls" or extension == ".xlsx":
                df = pd.read_excel(filename)
                return render_template("files_stat.html", tables=[df.head(10).to_html(classes='data', index = False, justify = 'left')])
            else:
                return render_template("files_stat.html", message = "Erreur de lecture du fichier")
        else:
            return render_template("files_stat.html", message = "Fichier ou extension inconnu !")
    return render_template("files_stat.html")

#Page pour le modèle de ML
@MyApp.route("/digits", methods=["GET", "POST"])
def digits():
    if request.method == 'POST':
        action = request.form['action']
        if action == "train":
            #Entrainement d'un nouveau modèle
            #Chargement du jeu de données
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
            #Preprocessing
            X_train = X_train.reshape(X_train.shape[0],-1)/255 #division par 255 pour avoir des valeurs entre 0 et 1
            X_test = X_test.reshape(X_test.shape[0],-1)/255
            X_train = X_train.astype(np.float32)
            X_test = X_test.astype(np.float32)
            #Entrainement du modèle
            regressor = LogisticRegression(solver='lbfgs', max_iter=1000, n_jobs=8)
            regressor.fit(X_train, y_train)
            #Prédiction et évaluation
            y_pred = regressor.predict(X_test)
            score = regressor.score(X_test, y_test)
            print(f"{score=}")
            #Sauvegarde sur disque pour réutilisation
            save = open("./__private/model.pkl", "wb")
            pickle.dump(regressor, save)
            save.close()
            return render_template("ml_digits.html")
        elif action == "predict_img":
            #On vérifie que l'utilisateur a bien sélectionné une image
            if 'image' not in request.files:
                flash("Veuillez choisir une image !")
                return render_template("ml_digits.html", message = "OK")
            image = request.files['image']
            print(image)
            if image:
                img = Image.open(image.stream)
                with BytesIO() as buf:
                    img.save(buf, 'jpeg')
                    displayable = base64.b64encode(buf.getvalue()).decode()
                    predictible = np.array(img).reshape(1,-1)
                    regressor = pickle.load(open("./__private/model.pkl", "rb"))
                    prediction = regressor.predict(predictible)
                    proba = regressor.predict_proba(predictible)
                return render_template("ml_digits.html", message = "OK", objects = (displayable, prediction[0], proba[0,prediction[0]] * 100))
            return render_template("ml_digits.html", message = "OK")
    #Affichage initial de la page : a-t-on déjà un modèle entrainé ?
    if os.path.exists("./__private/model.pkl"):
        flash("Modèle chargé, prêt à être utilisé !")
        return render_template("ml_digits.html", message = "OK")
    else: #On entraine un nouveau modèle
        return render_template("ml_digits.html")

#Route dédiée à la reconnaissance d'image à partir d'un dessin de l'utilisateur
@MyApp.route("/canvas_predict", methods=["POST"])
def get_image():
    image_b64 = request.values['imageBase64']
    image_data = re.sub('^data:image/.+;base64,', '', image_b64)
    image_data = base64.b64decode(image_data)
    with open("./__private/out.png", "wb") as out_file:
        out_file.write(image_data)

    img = cv2.imread('./__private/out.png', cv2.IMREAD_UNCHANGED)
    #Convert to gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Resize image to fit the model
    scale_percent = 16 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    resized = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
    cv2.imwrite('./__private/resized.png', resized)
    image_np = np.array(resized).reshape(1,-1)
    regressor = pickle.load(open("./__private/model.pkl", "rb"))
    prediction = regressor.predict(image_np)[0]
    proba = regressor.predict_proba(image_np)
    print(f"{prediction=}, proba={proba[0,prediction]}")
    return dumps({ "prediction" : str(prediction), "proba": str(proba[0,prediction] * 100) })

if __name__ == "__main__":
    MyApp.run(debug=True)
