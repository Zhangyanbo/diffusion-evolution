import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="diffevo",
    version="0.2.3",
    author="Yanbo Zhang",
    author_email="yzhang86@tufts.edu",
    description="Diffusion Evolutionary Algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Zhangyanbo/diffevo",
    project_urls={
        "Bug Tracker": "https://github.com/Zhangyanbo/diffevo/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=['diffevo'],
    python_requires=">=3.6",
)