# tensorflow_mnist_sru
A naive tensorflow mnist SRU example, speed is not optimized.

# usage
```
python train.py SRU
```
or
```
python train.py LSTM
```

# train log
```
--------------
LSTM network

[21:29:29.047] Epoch[1/3] Step[1/429] Train Minibatch Loss= 6.7037, Training Accuracy= 0.1484
[21:29:32.328] Epoch[1/3] Step[100/429] Train Minibatch Loss= 0.1882, Training Accuracy= 0.9453
[21:29:35.656] Epoch[1/3] Step[200/429] Train Minibatch Loss= 0.1174, Training Accuracy= 0.9688
[21:29:39.065] Epoch[1/3] Step[300/429] Train Minibatch Loss= 0.0988, Training Accuracy= 0.9609
[21:29:42.432] Epoch[1/3] Step[400/429] Train Minibatch Loss= 0.0790, Training Accuracy= 0.9766
[21:29:43.556] Epoch[2/3] Step[1/429] Train Minibatch Loss= 0.0739, Training Accuracy= 0.9766
[21:29:46.850] Epoch[2/3] Step[100/429] Train Minibatch Loss= 0.1305, Training Accuracy= 0.9609
[21:29:50.155] Epoch[2/3] Step[200/429] Train Minibatch Loss= 0.0396, Training Accuracy= 0.9922
[21:29:53.420] Epoch[2/3] Step[300/429] Train Minibatch Loss= 0.0611, Training Accuracy= 0.9688
[21:29:56.757] Epoch[2/3] Step[400/429] Train Minibatch Loss= 0.0499, Training Accuracy= 0.9766
[21:29:57.765] Epoch[3/3] Step[1/429] Train Minibatch Loss= 0.0275, Training Accuracy= 0.9922
[21:30:01.142] Epoch[3/3] Step[100/429] Train Minibatch Loss= 0.0119, Training Accuracy= 1.0000
[21:30:04.452] Epoch[3/3] Step[200/429] Train Minibatch Loss= 0.0285, Training Accuracy= 1.0000
[21:30:07.776] Epoch[3/3] Step[300/429] Train Minibatch Loss= 0.0105, Training Accuracy= 1.0000
[21:30:11.112] Epoch[3/3] Step[400/429] Train Minibatch Loss= 0.0546, Training Accuracy= 0.9766
[21:30:12.083] Optimization Finished!
[21:30:12.083] save model to model/mnist_nn.
[21:30:12.816] try to load model from model/mnist_nn.
[21:30:12.871] load model success
[21:30:13.947] Testing Accuracy: 0.980368614197

-------------
SRU network

[21:28:22.461] Epoch[1/3] Step[1/429] Train Minibatch Loss= 2.2775, Training Accuracy= 0.2344
[21:28:25.181] Epoch[1/3] Step[100/429] Train Minibatch Loss= 0.5900, Training Accuracy= 0.7812
[21:28:27.940] Epoch[1/3] Step[200/429] Train Minibatch Loss= 0.3459, Training Accuracy= 0.8906
[21:28:30.782] Epoch[1/3] Step[300/429] Train Minibatch Loss= 0.1854, Training Accuracy= 0.9609
[21:28:33.651] Epoch[1/3] Step[400/429] Train Minibatch Loss= 0.1230, Training Accuracy= 0.9531
[21:28:34.636] Epoch[2/3] Step[1/429] Train Minibatch Loss= 0.1979, Training Accuracy= 0.9453
[21:28:37.388] Epoch[2/3] Step[100/429] Train Minibatch Loss= 0.1033, Training Accuracy= 0.9688
[21:28:40.225] Epoch[2/3] Step[200/429] Train Minibatch Loss= 0.1192, Training Accuracy= 0.9609
[21:28:43.135] Epoch[2/3] Step[300/429] Train Minibatch Loss= 0.0294, Training Accuracy= 0.9922
[21:28:46.066] Epoch[2/3] Step[400/429] Train Minibatch Loss= 0.0988, Training Accuracy= 0.9766
[21:28:46.929] Epoch[3/3] Step[1/429] Train Minibatch Loss= 0.0817, Training Accuracy= 0.9922
[21:28:49.824] Epoch[3/3] Step[100/429] Train Minibatch Loss= 0.0673, Training Accuracy= 0.9922
[21:28:52.707] Epoch[3/3] Step[200/429] Train Minibatch Loss= 0.0836, Training Accuracy= 0.9766
[21:28:55.718] Epoch[3/3] Step[300/429] Train Minibatch Loss= 0.0565, Training Accuracy= 0.9844
[21:28:58.637] Epoch[3/3] Step[400/429] Train Minibatch Loss= 0.0340, Training Accuracy= 0.9922
[21:28:59.511] Optimization Finished!
[21:28:59.511] save model to model/mnist_nn.
[21:29:00.364] try to load model from model/mnist_nn.
[21:29:00.428] load model success
[21:29:01.153] Testing Accuracy: 0.97185498476
```
