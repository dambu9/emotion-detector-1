# Emotion Detector

Using Flask to build a Restful API Server.

Integration with Flask-SQLalchemy.

### Extension:
- SQL ORM: [Flask-SQLalchemy](http://flask-sqlalchemy.pocoo.org/2.1/)

## Installation

Install with pip:

```
$ pip install -r requirements.txt
```

## Flask Application Structure 
```
.
|──────app/
| |────__init__.py
| |────api/
| | |────__init__.py
| |────templates/
| |────model/
| |────static/
|──────routes.py

```


## Flask Configuration

#### Example

```
app = Flask(__name__)
app.config['DEBUG'] = True
```
 
## Run Flask
### Run flask for develop
```
$ python routes.py
```
In flask, Default port is `5000`

Emotion Detector document page:  `http://127.0.0.1:5000/`

## Reference

Offical Website

- [Flask](http://flask.pocoo.org/)
- [Flask Extension](http://flask.pocoo.org/extensions/)
- [Flask restplus](http://flask-restplus.readthedocs.io/en/stable/)
- [Flask-SQLalchemy](http://flask-sqlalchemy.pocoo.org/2.1/)
- [Flask-OAuth](https://pythonhosted.org/Flask-OAuth/)
- [elasticsearch-dsl](http://elasticsearch-dsl.readthedocs.io/en/latest/index.html)
- [gunicorn](http://gunicorn.org/)

Tutorial

- [Flask Overview](https://www.slideshare.net/maxcnunes1/flask-python-16299282)
- [In Flask we trust](http://igordavydenko.com/talks/ua-pycon-2012.pdf)

[Wiki Page](https://github.com/tsungtwu/flask-example/wiki)