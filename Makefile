SHELL := /bin/sh

install:
	pip install -r requirements.txt

test:
	pytest
