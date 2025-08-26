from setuptools import setup, find_packages

setup(
    name="doodle-recognizer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'fastapi==0.95.2',
        'uvicorn==0.22.0',
        'python-multipart==0.0.6',
        'numpy==1.24.3',
        'Pillow==9.5.0',
        'python-dotenv==1.0.0',
        'tensorflow-cpu==2.12.0',
        'pydantic==1.10.7',
        'python-jose[cryptography]==3.3.0',
        'passlib[bcrypt]==1.7.4',
        'httpx==0.24.0',
    ],
    python_requires='>=3.8,<3.11',
)
