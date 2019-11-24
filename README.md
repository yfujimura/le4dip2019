# 2019年度計算機科学実験4

## 仕様
- Network の定義
	- net = Sequential(Affine(784,100), Sigmoid(), ...)
- Optimizer の定義
	- optimizer = SGD(net.getLayers(), lr=0.01)
- 学習方法
	- forward: loss = net(X, target=Y) # X, Y はミニバッチ
	- 勾配の計算: net.backprop(loss)
	- パラメータ更新: optimizer.step()
- 推論
	- net.(X)
- モデルの保存
	- net.saveParams("params")
- モデルの読み込み
	- net.loadParams("params.npy")

## 実験結果
- MNIST
	- (784,100)-sigmoid-(100,10)-softmax
		- SGD:lr=0.01
			- epochs=50
			- accuracy=0.9236
			- 学習済みパラメータ: 50.npy
	- (784,100)-relu-(100,10)-softmax
		- SGD:lr=0.01	
			- epochs=50
			- accuracy=0.9595
			- 学習済みパラメータ: relu_50.npy
	- (784,100)-relu-dropout(0.5)-(100,10)-softmax
		- SGD:lr=0.01
			-epochs=50
			- accuracy=0.9401
			- 学習済みパラメータ: relu_do_50.npy
	- (784,100)-batchnorm-relu-(100,10)-softmax
		- SGD:lr=0.01
			- epochs=50
			- accuracy=0.9697
			- 学習済みパラメータ: relu_bn_50.npy
		- MomentumSGD:lr=0.01 alpha=0.9
			- epochs=50
			- accuracy=0.9759
			- 学習済みパラメータ: relu_bn_50_msgd.npy
		- AdaGrad:lr=0.001 h0=1.0e-8 
			- epochs=50
			- accuracy=0.9532
			- 学習済みパラメータ: relu_bn_50_adagrad.npy
