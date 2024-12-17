PACKAGE := cellcontroller

build:
	nuitka --lto=yes --standalone --onefile -o $(PACKAGE) --include-data-dir=./objects=objects main.py 
	upx --best $(PACKAGE)

install:
	@poetry install
	@echo "Successfully installed $(PACKAGE) locally"

install-global:
	@poetry build -f wheel -o . > /dev/null
	@pip uninstall $(PACKAGE) -y --root-user-action=ignore > /dev/null
	@pip install *.whl --root-user-action=ignore > /dev/null
	@rm *.whl
	@echo "Successfully installed $(PACKAGE) globally"

rehash:
	@poetry lock --no-update
	@poetry update

# test-cov:
# 	@poetry run pytest --cov=$(PACKAGE) --cov-report=term-missing