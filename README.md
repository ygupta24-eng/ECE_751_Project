# Running on Ubuntu with pyenv:

## Install Dependencies:
~~~
sudo apt update && sudo apt install make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl git libncursesw5-dev \
xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
~~~

## Install pyenv:
~~~
curl -fsSL https://pyenv.run | bash
~~~

### Run the following on bash:
~~~
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc
~~~

### Reload shell:
~~~
exec "$SHELL"
~~~

## Install Python 3.8.20
~~~
pyenv install 3.8.20
~~~

### Verify installation using
~~~
pyenv versions
~~~

## Create a virtual environment: 
### Navigate to the main folder and run
~~~
pyenv virtualenv 3.8.20 venv
pyenv activate venv
~~~

## Install all the dependencies for EcoWild
~~~
pip install -r requirements.txt
~~~

## Run the script:
~~~
python3 inference_main.py config/config_setup_0.90036452TP_0.58399005FP_1dayReservedEnergy.json
~~~

#### The output folder /Inference will contain graphs.
