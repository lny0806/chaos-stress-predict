# chaos-stress-predict
The implementation of "Stress Prediction based on Chaos Theory and an Event-Behavior-Stress Triangle Model" using pytorch
## Getting Start

### Install
We provide a pytorch implementation. The following are needed:
* torch==1.7.1
* tensorflow==2.3.0

## Stress Prediction
The training settings can be found in ``config.py``, including parameter settings. Modifing configuration and then start training and testing:

For 2-label prediction, set ``self.num_classes=2`` in ``config.py`` and run:
```
python main-two.py
```

For 3-label prediction, set ``self.num_classes=3`` in ``config.py`` and run:
```
python main-three.py
```
