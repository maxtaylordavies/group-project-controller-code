from setuptools import setup

setup(
    name="group-project-controller-code",
    version="0.0.1",
    install_requires=[
        "gym==0.26.0",
        "pygame==2.1.0",
        "torch",
        "tqdm",
        "opencv-python",
        "matplotlib",
        "numpy",
    ],
    packages=["src"],
)
