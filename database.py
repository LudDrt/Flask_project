import mysql.connector as mysql
import os


def get_db():
    db = mysql.connect(host = "localhost", user = "root", password = "root", database = "little_flask")
    return db
