toy:
	test -f venv/bin/activate || virtualenv -p $(shell which python) venv
	. venv/bin/activate ; \
		pip install numpy scipy sklearn ; \
		python model.py -t ../../data/train_toy/ -u ../../data/test_toy/ -m mean -c 0 -v > ../../output/toy-results.txt

example:
	test -f venv/bin/activate || virtualenv -p $(shell which python) venv
	. venv/bin/activate ; \
		pip install numpy scipy sklearn ; \
		python model.py -h

clean:
	rm -rf venv
		
streamspot_test:
	test -f venv/bin/activate || virtualenv -p $(shell which python) venv
	. venv/bin/activate ; \
		pip install numpy scipy scikit-learn ; \
		python model.py -t ../../data/train_toy/ -u ../../data/test_toy/ -m mean -c 5 -v > ../../output/streamspot-results.txt
		
evasion_mimicry_org_batch:
	test -f venv/bin/activate || virtualenv -p $(shell which python) venv
	. venv/bin/activate ; \
		pip install numpy scipy scikit-learn ; \
		python model.py -t ../../data/train_mimicry_evasion_org_batch/ -u ../../data/test_mimicry_evasion_org_batch/ -m mean -c 1 -v > ../../output/mimicry-evasion-results-org-batch.txt
	
evasion_mimicry:
	test -f venv/bin/activate || virtualenv -p $(shell which python) venv
	. venv/bin/activate ; \
		pip install numpy scipy scikit-learn ; \
		python model.py -t ../../data/train_mimicry_evasion/ -u ../../data/test_mimicry_evasion/ -m mean -c 1 -v > ../../output/mimicry-evasion-results.txt

attack_mimicry:
	test -f venv/bin/activate || virtualenv -p $(shell which python) venv
	. venv/bin/activate ; \
		pip install numpy scipy scikit-learn ; \
		python model.py -t ../../data/train_mimicry_evasion/ -u ../../data/test_mimicry_attack/ -m mean -c 1 -v > ../../output/mimicry-attack-results.txt

benign_mimicry:
	test -f venv/bin/activate || virtualenv -p $(shell which python) venv
	. venv/bin/activate ; \
		pip install numpy scipy scikit-learn ; \
		python model.py -t ../../data/train_mimicry_evasion/ -u ../../data/test_mimicry_benign/ -m mean -c 1 -v > ../../output/mimicry-benign-results.txt
