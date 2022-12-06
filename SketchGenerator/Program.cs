﻿using SketchGenerator;
using TorchSharp;
using DeepConvGAN;

//Parametrize pipeline
const int numEpoch = 5;
const int batchSize = 128;
const int noiseDimensions = 50;
const double optimizersLr = 0.0002;
(double, double) optimizersBetas = (0.5, 0.999);


//Create models
//NetworksTrainer networksTrainer = new NetworksTrainer(DeviceType.CUDA, noiseDimensions, imageOutputSize, optimizersLr, optimizersBetas);
//networksTrainer.Train(numEpoch, batchSize);


var n = new NetworkTrainer(optimizersBetas, numEpoch, batchSize, noiseDimensions, optimizersLr);
