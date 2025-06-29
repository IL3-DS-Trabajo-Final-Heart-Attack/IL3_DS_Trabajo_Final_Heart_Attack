all: preprocess train evaluate

preprocess:
	conda run -n heart-attack-minimal python src/preprocess.py

train:
	conda run -n heart-attack-minimal python src/train.py

evaluate:
	conda run -n heart-attack-minimal python src/evaluate.py

clean:
	rm -rf data/processed/*
	rm -rf models/*