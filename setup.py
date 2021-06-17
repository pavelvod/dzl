from setuptools import setup


def get_requirements():
    """
    lists the requirements to install.
    """
    requirements = []
    try:
        with open('requirements.txt') as f:
            requirements = f.read().splitlines()
    except Exception as ex:
        with open('DecoraterBotUtils.egg-info\requires.txt') as f:
            requirements = f.read().splitlines()
    return requirements


setup(
    name='dzl',
    version='0.3',
    packages=['dzl'],
    url='',
    license='',
    author='Pavel Vodolazov',
    author_email='pavel.vod1@gmail.com',
    install_requires=get_requirements(),
    description=''
)
