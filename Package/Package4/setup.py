from setuptools import setup
from draw_circle_fourier import __version__ as current_version

setup(
  name='draw_circle_fourier',
  version=current_version,
  description='Drawing spinning circle with Fourier',
  url='https://github.com/Paul30hub/Fourier-transform-drawing',
  author='Burgat Paul, Cordoval Chloë, Guillaumont Pierre, Koan Kenjy',
  author_email='paul.burgat@etu.umontpellier.fr, chloe.cordoval@etu.umontpellier.fr, pierre.guillaumont@etu.umontpellier.fr, kenjy.koan@etu.umontpellier.fr',
  license='MIT',
  packages=['draw_circle_fourier','draw_circle_fourier.CoeffFourier', 'biketrauma.ImageAnimation', 'biketrauma.ImageReader'],
  zip_safe=False
)
