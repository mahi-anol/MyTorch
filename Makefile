module0:
	./Scripts/create_m0_dir.sh && cd minitorch-module0 && exec bash
requirement:
	pip install -r requirements.txt
.PHONY: module0