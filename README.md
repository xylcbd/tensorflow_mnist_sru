# tensorflow_mnist_sru
a naive tensorflow mnist SRU example, speed is not optimized.

# usage
```python
python train.py
```

# train log
```
--------------
LSTM network
[21:46:23.678] Epoch[1/3] Step[100/429] Train Minibatch Loss= 0.3628, Training Accuracy= 0.9141
[21:46:27.319] Epoch[1/3] Step[200/429] Train Minibatch Loss= 0.2073, Training Accuracy= 0.9453
[21:46:30.858] Epoch[1/3] Step[300/429] Train Minibatch Loss= 0.1160, Training Accuracy= 0.9688
[21:46:34.523] Epoch[1/3] Step[400/429] Train Minibatch Loss= 0.0621, Training Accuracy= 0.9844
[21:46:35.725] Epoch[2/3] Step[1/429] Train Minibatch Loss= 0.0962, Training Accuracy= 0.9688
[21:46:39.163] Epoch[2/3] Step[100/429] Train Minibatch Loss= 0.0465, Training Accuracy= 0.9844
[21:46:42.485] Epoch[2/3] Step[200/429] Train Minibatch Loss= 0.1000, Training Accuracy= 0.9688
[21:46:46.020] Epoch[2/3] Step[300/429] Train Minibatch Loss= 0.0493, Training Accuracy= 0.9844
[21:46:49.353] Epoch[2/3] Step[400/429] Train Minibatch Loss= 0.0401, Training Accuracy= 0.9922
[21:46:50.502] Epoch[3/3] Step[1/429] Train Minibatch Loss= 0.0323, Training Accuracy= 0.9922
[21:46:54.046] Epoch[3/3] Step[100/429] Train Minibatch Loss= 0.0369, Training Accuracy= 0.9844
[21:46:57.674] Epoch[3/3] Step[200/429] Train Minibatch Loss= 0.0484, Training Accuracy= 0.9844
[21:47:01.186] Epoch[3/3] Step[300/429] Train Minibatch Loss= 0.0522, Training Accuracy= 0.9922
[21:47:04.735] Epoch[3/3] Step[400/429] Train Minibatch Loss= 0.0210, Training Accuracy= 0.9922
[21:47:05.826] Optimization Finished!
[21:47:05.826] save model to model/mnist_nn.
[21:47:08.348] try to load model from model/mnist_nn.
[21:47:08.383] load model success
[21:47:09.421] Testing Accuracy: 0.981470346451

-------------
SRU network
[21:47:52.253] Epoch[1/3] Step[100/429] Train Minibatch Loss= 0.6344, Training Accuracy= 0.7734
[21:47:55.190] Epoch[1/3] Step[200/429] Train Minibatch Loss= 0.4133, Training Accuracy= 0.8594
[21:47:57.931] Epoch[1/3] Step[300/429] Train Minibatch Loss= 0.1899, Training Accuracy= 0.9531
[21:48:00.635] Epoch[1/3] Step[400/429] Train Minibatch Loss= 0.2103, Training Accuracy= 0.9141
[21:48:01.572] Epoch[2/3] Step[1/429] Train Minibatch Loss= 0.2073, Training Accuracy= 0.9297
[21:48:04.422] Epoch[2/3] Step[100/429] Train Minibatch Loss= 0.1949, Training Accuracy= 0.9453
[21:48:07.419] Epoch[2/3] Step[200/429] Train Minibatch Loss= 0.1502, Training Accuracy= 0.9453
[21:48:10.180] Epoch[2/3] Step[300/429] Train Minibatch Loss= 0.0990, Training Accuracy= 0.9453
[21:48:13.021] Epoch[2/3] Step[400/429] Train Minibatch Loss= 0.0821, Training Accuracy= 0.9766
[21:48:13.984] Epoch[3/3] Step[1/429] Train Minibatch Loss= 0.0748, Training Accuracy= 0.9688
[21:48:16.779] Epoch[3/3] Step[100/429] Train Minibatch Loss= 0.0592, Training Accuracy= 0.9844
[21:48:19.662] Epoch[3/3] Step[200/429] Train Minibatch Loss= 0.0475, Training Accuracy= 0.9844
[21:48:22.530] Epoch[3/3] Step[300/429] Train Minibatch Loss= 0.1152, Training Accuracy= 0.9766
[21:48:25.464] Epoch[3/3] Step[400/429] Train Minibatch Loss= 0.0876, Training Accuracy= 0.9688
[21:48:26.257] Optimization Finished!
[21:48:26.257] save model to model/mnist_nn.
[21:48:28.728] try to load model from model/mnist_nn.
[21:48:28.760] load model success
[21:48:29.499] Testing Accuracy: 0.964342951775
```
