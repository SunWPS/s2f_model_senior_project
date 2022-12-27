from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


# Function that initializes the db and creates the tables
def db_init(app):
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///data.db"
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False
    db.init_app(app)

    # Creates the logs tables if the db doesnt already exist
    with app.app_context():
        db.create_all()