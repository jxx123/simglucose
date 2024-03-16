rm -r build dist simglucose.egg-info
python setup.py sdist bdist_wheel
# python -m build
twine upload dist/* -r jxx123