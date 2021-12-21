import setuptools

scripts = ["./bin/prepare_maps"]

setuptools.setup(
    name="cmbsky",
    version="0.0.1",
    author="Niall MacCrann",
    author_email="nm746@cam.ac.uk",
    description="Tools for working with microwave sky sims",
    packages=["cmbsky"],
    include_package_data=True,
    package_data={'cmbsky': ['defaults.yaml']},
    scripts=scripts
)
