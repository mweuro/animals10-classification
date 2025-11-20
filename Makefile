.PHONY: all download split

download:
    bash scripts/download.sh

split:
    python3 src/split.py

download_and_split: 
	$(MAKE) download
	$(MAKE) split