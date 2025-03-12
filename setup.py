import setuptools




__version__ = "0.0.0"

REPO_NAME = "HNSW"
AUTHOR_USER_NAME = "bishwajyotichaudhary46"
SRC_REPO = "HNSW"
AUTHOR_EMAIL = "bishwajyotichaudhary46@gmail.com"



setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Vector Database",
    long_description="Storing Embedding",
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "component"},
    packages=setuptools.find_packages(where="component")
)