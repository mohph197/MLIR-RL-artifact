#!/bin/bash

export CONFIG_FILE_PATH=config/example.json
poetry install
poetry run train
