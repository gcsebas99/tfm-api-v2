#!/bin/bash
gunicorn -w 1 --threads 1 -t 120 -b 0.0.0.0:10000 app:app