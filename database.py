import mysql.connector as mysql

def get_db():
    db = mysql.connect(host = "localhost", user = "root", password = "root", database = "little_flask")
    return db
