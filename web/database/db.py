from flask_sqlalchemy import SQLAlchemy
import pymysql
db = SQLAlchemy()


# Function that initializes the db and creates the tables
def db_init(app):
    app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://root:mypassword@localhost/data"
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False
    db.init_app(app)

    # Creates the logs tables if the db doesnt already exist
    with app.app_context():
        db.create_all()