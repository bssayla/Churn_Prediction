from setuptools import setup, find_packages

setup(
    name='data_preprocessing',
    version='0.1',
    author='Ouaicha Mohamed',
    author_email='Ouaicha47@gmail.com',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'pytest',
        'scipy',
        'joblib',
        'streamlit'
    ]
)