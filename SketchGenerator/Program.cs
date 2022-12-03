using SketchGenerator;
using TorchSharp;

//Parametrize pipeline
const int numEpoch = 50;
const int batchSize = 128;
const long noiseDimensions = 50;
const long imageOutputSize = 28 * 28 * 1;
const double optimizersLr = 0.0002;
const (double, double) optimizersBetas = (0.5, 0.999);

//Create models
NetworksTrainer networksTrainer = new NetworksTrainer(DeviceType.CUDA, noiseDimensions, imageOutputSize, optimizersLr, optimizersBetas);
networksTrainer.Train(numEpoch, batchSize);