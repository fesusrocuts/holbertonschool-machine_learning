# 0x00. Linear Algebra
## General
What is a vector?
What is a matrix?
What is a transpose?
What is the shape of a matrix?
What is an axis?
What is a slice?
How do you slice a vector/matrix?
What are element-wise operations?
How do you concatenate vectors/matrices?
What is the dot product?
What is matrix multiplication?
What is Numpy?
What is parallelization and why is it important?
What is broadcasting?
## Requirements
Python Scripts
Allowed editors: vi, vim, emacs
All your files will be interpreted/compiled on Ubuntu 16.04 LTS using python3 (version 3.5)
Your files will be executed with numpy (version 1.15)
All your files should end with a new line
The first line of all your files should be exactly #!/usr/bin/env python3
A README.md file, at the root of the folder of the project, is mandatory
Your code should follow pycodestyle (version 2.5)
All your modules should have documentation (python3 -c 'print(__import__("my_module").__doc__)')
All your classes should have documentation (python3 -c 'print(__import__("my_module").MyClass.__doc__)')
All your functions (inside and outside a class) should have documentation (python3 -c 'print(__import__("my_module").my_function.__doc__)' and python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)')
Unless otherwise noted, you are not allowed to import any module
All your files must be executable
The length of your files will be tested using wc

### Installing Ubuntu 16.04 and Python 3.5

### Installing pip 19.1
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
rm get-pip.py
To check that pip has been successfully downloaded, use pip -V. Your output should look like:

$ pip -V
pip 19.1.1 from /usr/local/lib/python3.5/dist-packages/pip (python 3.5)

### Installing numpy 1.15, scipy 1.3, and pycodestyle 2.5
$ pip install --user numpy==1.15
$ pip install --user scipy==1.3
$ pip install --user pycodestyle==2.5
