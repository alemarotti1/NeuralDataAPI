from setuptools import setup

setup(
    name="NeuralDataAPI",
    version="0.0.1",
    description="This library offers the necessary functionality to interact with neural data using the kwik format.",
    py_modules=["NeuralDataAPI"],
    package_dir={"": "src"},
    install_requires=[
        "klusta"
    ]
)