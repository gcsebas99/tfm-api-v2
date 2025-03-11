#!/bin/bash
gunicorn -w 4 -t 120 -b 0.0.0.0:10000 app:app