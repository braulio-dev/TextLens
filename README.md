# TextLens
TextLens is a machine learning project that aims to parse presentation 
slides and extract the text from them. After all slides are collected
and processed, the text is then used to generate a summary of the entire
presentation in a markdown file.

## Environment Setup
Tesseract is required for the OCR functionality. To install Tesseract on your Machine, please
refer to the [Tesseract GitHub page](https://github.com/tesseract-ocr/tesseract).

Anaconda is recommended for managing the environment. To create the environment, run the following commands:
```bash
conda create --name textlens python=3.12.4
conda activate textlens
conda install -c conda-forge opencv pytesseract pytorch
pip install -r requirements.txt
```