a:
	@echo "Activating virtual environment..."
	@source venv/bin/activate

d:
	@echo "Deactivating virtual environment..."
	@deactivate


build:
	@echo "Building the project..."
	@rm -rf venv
	@python3 -m venv venv
	@. venv/bin/activate && pip install -r requirements.txt
	@python3 -m setup.py develop


t:
	@echo "Running fast tests..."
	@pytest tests -vv -m "not slow"
ta:
	@echo "Running all tests..."
	@pytest tests -vv
ts:
	@echo "Running slow tests..."
	@pytest tests -vv -m "slow"