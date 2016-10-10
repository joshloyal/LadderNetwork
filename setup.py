from setuptools import setup


PACKAGES = [
    'ladder_network',
    'ladder_network.ops',
]

def setup_package():
    setup(
        name="LadderNetwork",
        version='0.1.0',
        description="TensorFlow implementation of Rasmus et. al's Ladder Network",
        author='Joshua D. Loyal',
        url='https://github.com/joshloyal/LadderNetwork',
        license='MIT',
        install_requires=['numpy', 'tensorflow'],
        packages=PACKAGES,
    )


if __name__ == '__main__':
    setup_package()
