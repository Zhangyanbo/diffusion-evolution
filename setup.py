import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="diffevo",
    version="1.0.1",
    author="Yanbo Zhang",
    author_email="zhangybspm@gmail.com",
    description="Diffusion Evolutionary Algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Zhangyanbo/diffusion-evolution",
    project_urls={
        "Bug Tracker": "https://github.com/Zhangyanbo/diffusion-evolution/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    packages=['diffevo'],
    python_requires=">=3.6",
)