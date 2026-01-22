python mount/MaskDINO_test/mask.py \
	--config-file /workspace/MaskDINO/configs/ade20k/semantic-segmentation/maskdino_R50_bs16_160k_steplr.yaml \
	--input input/*  \
	--output mount/MaskDINO_test/MaskDINO_test_out \
	--opts MODEL.WEIGHTS MaskDINO-ADE20K.pth
