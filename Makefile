.PHONY:

train:
	@python train_binaural.py

submit:
	@qsub qsub.pbs

viz:
	@python visualize_locata.py

viz-multi:
	@python visualize_tau.py