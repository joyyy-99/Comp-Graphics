# Student Data Processor

## Overview

This project is designed to process and analyze student data from Excel sheets, specifically focusing on:
- **Generating unique email addresses** for each student.
- **Classifying students by gender** and creating separate lists for male and female students.
- **Computing name similarity** between male and female students using the LaBSE (Language-Agnostic BERT Sentence Embedding) model.
- **Saving results** in multiple formats such as CSV, TSV, JSON, and logs.

The project is written in Python and makes use of popular libraries such as `pandas` for data manipulation and `transformers` for natural language processing.

## Features

1. **Email Generation**: Generates email addresses for students based on their first initial and last name.
2. **Gender Classification**: Separates students into male and female categories and saves them as individual CSV files.
3. **Name Similarity Matrix**: Uses the LaBSE model to compute a similarity matrix between male and female names.
4. **File Saving**: Saves data in CSV, TSV, and JSON formats.
5. **Logging**: Tracks the number of male and female students, and logs special characters in student names.


## Requirements
- Python 3.7 or later
- Required Python packages:
  - pandas
  - openpyxl
  - transformers
  - torch

You can install the required dependencies by running:
bash
pip install pandas openpyxl transformers torch


##to run

python main.py

