import os
from setuptools import setup, find_packages


def package_files():
    cwd = os.getcwd()
    paths = [
        os.path.join(cwd, 'sic42', 'deathmatch_config.json')
    ]
    baselines_path = os.path.join(cwd, 'sic42', 'competitors')
    icons_path = os.path.join(cwd, 'sic42', 'icons')
    for fn in os.listdir(baselines_path):
        paths.append(os.path.join(baselines_path, fn))
    for fn in os.listdir(icons_path):
        paths.append(os.path.join(icons_path, fn))
    return paths

setup(
    name='sic42',
    version='0.3.0',
    description='Lab42 Swarm Intelligence Cup',
    url='https://github.com/lab42-global/sic42',
    author='Lab42',
    author_email='swarm@lab42.global',
    license='Apache License 2.0',
    packages=find_packages('.'),
    package_data={'': package_files()},
    include_package_data=True,
    install_requires=[
        'tqdm',
        'numpy',
        'pandas',
        'matplotlib',
        'pygame',
        'ffmpeg-python'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
        'Natural Language :: English'
    ],
)