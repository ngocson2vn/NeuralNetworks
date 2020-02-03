# Notes

## Training script
If you execute the training script `train_ann.py` on your PC, you may get 3 curves of Maximum loss, Average loss, and Maximum loss with different shapes than mine. The reason is that initial weights and biases are random values. So to save your time, I provided you with the second training script `train_ann_stable.py`. In this script, I have hardcoded initial weights and biases:
```python
# Preset weights and biases
n.w1 = -0.06958319062984783
n.w2 = -0.13036519749456643
n.w3 = -1.0571412909409637
n.w4 = 0.135581608179992
n.w5 = 0.5626741222756912
n.w6 = -0.010929699015683102
n.b1 = -0.83347780239002
n.b2 = 0.27420776870407937
n.b3 = 0.6237767133439941
```

If you execute `train_ann_stable.py`, I ensure that finally, you will get the following Figure:
![image](https://user-images.githubusercontent.com/1695690/73621946-5dfced00-467b-11ea-98ee-ff29943fe11a.png)
