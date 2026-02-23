MyTorch:
	Scripts/create_m0_dir.sh && . .venv/bin/activate && pip install -r requirements.txt && exec bash
create-env:
	python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
.PHONY: MyTorch create-env