from setuptools import setup
import os

def get_long_description():
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md"),
        encoding="utf8",
    ) as fp:
        return fp.read()

VERSION = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plu", "VERSION")).read().strip()

setup(
    name="plu",
    description="audio-conditional language models are multi-task recognizers",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Vol Kyrylov",
    author_email="vol@wilab.org.ua",
    url="https://github.com/proger/plu",
    project_urls={
        "Issues": "https://github.com/proger/plu/issues",
        "CI": "https://github.com/proger/plu/actions",
        "Changelog": "https://github.com/proger/plu/releases",
    },
    license="Apache License, Version 2.0",
    version=VERSION,
    packages=["plu"],
    entry_points="""
        [console_scripts]
        +balance=plu.balance:main
        +soundcheck=plu.soundcheck:main
        +dataloader=plu.dataloader:main
    """,
    install_requires=["tiktoken", "openai-whisper", "soundfile", "transformers", "datasets"],
    python_requires=">=3.8",
    include_package_data=True,
    package_data={"plu": ["VERSION"]},
)
