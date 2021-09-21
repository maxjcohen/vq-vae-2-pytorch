epoch = 1
dataset = cifar10

# Train vqvae weights
checkpoint/vqvae.pt:
	python train_vqvae.py --epoch $(epoch)
vqvae: checkpoint/vqvae.pt

# Extract codes
$(dataset)/data.mdb:
	python extract_code.py --ckpt checkpoint/vqvae.pt --name $(dataset)
codes: $(dataset)/data.mdb vqvae

# Train pixelsnail top and bottom
checkpoint/pixelsnail_top.pt:
	python train_pixelsnail.py --epoch $(epoch) --hier top cifar10
checkpoint/pixelsnail_bottom.pt:
	python train_pixelsnail.py --epoch $(epoch) --hier bottom cifar10
prior: checkpoint/pixelsnail_bottom.pt checkpoint/pixelsnail_top.pt codes

# Sample from prior
sample.jpg: prior
	python sample.py --vqvae vqvae.pt --top pixelsnail_top.pt --bottom pixelsnail_bottom.pt sample.jpg
