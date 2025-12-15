from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="ktools",  # Replace with your project's name
    version="0.0.5",  # Initial release version
    author="Yuwei Zhu",  # Replace with your name or your organization's name
    author_email="yuweizhu29@gmail.com",  # Replace with your email
    description="A repository of tools to use for kaggle competitions",  # Replace with a short description of your project
    url="https://github.com/yuwei-1/Kaggle-tools",  # Replace with the URL to your project repository
    packages=find_packages(),  # Automatically find all packages in your project
    install_requires=required,  # Use the requirements.txt file to list dependencies
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",  # Replace with your chosen license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",  # Specify the Python version compatibility
)
