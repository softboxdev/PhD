from flask import Blueprint

bp = Blueprint('example', __name__)

@bp.route('/')
def home():
    return "Hello, Flask!"
