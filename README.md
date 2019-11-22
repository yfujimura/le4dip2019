# 2019年度計算機科学実験4

## 課題4
- (784,100)-sigmoid-(100,10)-softmax
- SGD:lr=0.01, epochs=50
- accuracy: 0.9221
- params_50.npy, training_loss_50.png

## 発展課題
- (784,100)-relu-(100,10)-softmax
	- SGD:lr=0.01, epochs=50
	- accuracy: 0.9585
	- params_relu_50.npy, training_loss_relu_50.png
- (784,100)-relu-dropout(0.5)-(100,10)-softmax
	- SGD:lr=0.01, epochs=50
	- accuracy: 0.9425
	- params_relu_do_50.npy, training_loss_relu_do_50.png
- (784,100)-batchnorm-relu-(100,10)-softmax
	- SGD:lr=0.01, epochs=50
	- accuracy: 0.9574
	- params_relu_bn_50.npy, training_loss_relu_bn_50.png
