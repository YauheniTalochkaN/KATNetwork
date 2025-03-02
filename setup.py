from setuptools import setup, find_packages

setup(
    name='KATNetwork',
    version='1.0.0',
    description='PyTorch-based module for KAT layers and KAT MLPs.',
    author='Yauheni Talochka',
    author_email='yauheni.talochka@gmail.com',
    url='https://github.com/YauheniTalochkaN/KATNetwork',
    license='CC BY-NC-ND 4.0',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'torch>=2.5.1',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: CC BY-NC-ND 4.0 License',
        'Programming Language :: Python :: 3.9',  
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',  
        'Programming Language :: Python :: 3.12',  
        'Programming Language :: Python :: 3.13',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.9',
)