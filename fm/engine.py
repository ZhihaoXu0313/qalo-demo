import os
import matplotlib.pyplot as plt 
import sys
import time

from arguments import JSONParameters
args = JSONParameters()

import xlearn as xl


class ffm:
    def __init__(self, lr=args.factor_machine.learning_rate, 
                 reg=args.factor_machine.reg_lambda, 
                 opt=args.factor_machine.opt,
                 k=args.factor_machine.latent_space,
                 epoch=args.factor_machine.epoch,
                 metric=args.factor_machine.metric):
        self.model = xl.create_ffm()
        self.param = {'task': 'reg', 'lr': lr, 'lambda': reg, 'k': k, 'opt': opt, 'epoch': epoch, 'metric': metric}
        
    def train(self, trainSet, validSet, model_txt, model_out, restart=False):
        t_start = time.time()
        self.model.setTrain(trainSet)
        self.model.setValidate(validSet)
        if restart and os.path.exists(model_out):
            self.model.setPreModel(model_out)
        self.model.setTXTModel(model_txt)
        self.model.disableNorm()
        self.model.fit(self.param, model_out)
        t_end = time.time()
        print("total ffm train wall time: " + str(t_end - t_start))

    def infer(self, testSet, model_out, testResult, fig=False):
        self.model.setTest(testSet)  # set test set for fm model.
        self.model.predict(model_out, testResult)
        e_pred = []
        with open(testResult, 'r') as f:
            for line in f:
                e_pred.append(float(line))
        e = []
        with open(testSet, 'r') as f:
            for line in f:
                e.append(float(line.strip().split(' ')[0]))
        if fig:
            plt.scatter(e, e_pred)
            plt.savefig("evaluation.png")
        return e_pred, e
