# Default target: run all analyses
.PHONY: all
all: analyses

# -------------------------------
# Clean: delete intermediate files but keep code and raw data
.PHONY: clean
clean:
#	@echo "Running python main.py clean"
	python3 main.py clean

# -------------------------------
# Run analyses (without downloading raw data or making current predictions)
.PHONY: analyses
analyses:
#	@echo "Running python main.py train"
	python3 main.py train

# -------------------------------
# Make current predictions
.PHONY: predictions
predictions:
#	@echo "Running python main.py predictions"
	@python3 main.py predictions

# -------------------------------
# Download raw data (delete existing raw data first)
.PHONY: rawdata
rawdata:
#	@echo "Running python main.py rawdata"
	python3 main.py rawdata
