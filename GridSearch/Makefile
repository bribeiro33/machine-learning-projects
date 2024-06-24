ALL_TESTS := $(wildcard test_files/*.test)
PYTHON := python3


all: 
	@echo "Testing $@..."
	$(PYTHON) -m unittest discover test_files


%.test:
	@echo "Testing grid_$@"
	$(PYTHON) -m unittest test_files/test_grid_$*.py

reset:
	rm -rf __pycache__
	rm -rf env