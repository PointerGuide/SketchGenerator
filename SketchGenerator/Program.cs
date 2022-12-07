using SketchGenerator;
using TorchSharp;
using DeepConvGAN;

//Parametrize pipeline
const int numEpoch = 50;
const int batchSize = 256;
const int noiseDimensions = 128;
const double optimizersLr = 0.0002;
(double, double) optimizersBetas = (0.5, 0.999);


//Create models
//NetworksTrainer networksTrainer = new NetworksTrainer(DeviceType.CUDA, noiseDimensions, imageOutputSize, optimizersLr, optimizersBetas);
//networksTrainer.Train(numEpoch, batchSize);

//Classes: airplane, apple, bicycle, hat, snowflake, ball
NetworkTrainer networkTrainer = new NetworkTrainer("ball", optimizersBetas, numEpoch, batchSize, noiseDimensions, optimizersLr);
networkTrainer.Train();
