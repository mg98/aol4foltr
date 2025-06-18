build:
	./run.sh

install:
	chmod +x run.sh
	pip install -r requirements.txt

# Clean target to remove temporary files and directories
clean:
	rm -rf indexes docs_jsonl slurm-*.out

.PHONY: install clean
