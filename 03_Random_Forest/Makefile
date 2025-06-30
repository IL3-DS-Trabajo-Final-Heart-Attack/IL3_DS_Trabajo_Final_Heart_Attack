all: preprocess train evaluate

preprocess:
	conda run -n rf_env python src/preprocess.py

train:
	conda run -n rf_env python src/train.py

evaluate:
	conda run -n rf_env python src/evaluate.py

clean:
	rm -rf data/processed/*
	rm -rf models/*

test_pacient:
	conda run -n rf_env python src/pacient_test.py