using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torchvision;
using NumSharp;

namespace DeepConvGAN
{
    public class NetworkTrainer
    {
        private readonly Tensor _dataset;
        private readonly TorchSharp.Modules.BCELoss _lossFunction;
        private readonly int _batchSize;
        private const int _realLabel = 1, _fakeLabel = 0;

        private readonly Generator _generator;
        private readonly Discriminator _discriminator;

        private readonly TorchSharp.Modules.Adam _generatorOptimizer;
        private readonly TorchSharp.Modules.Adam _discriminatorOptimizer;
        public NetworkTrainer((double, double) optimizerBetas, uint numEpoch = 5, int batchSize = 128, 
            int generatorInputSize = 50, double learningRate = 0.0002) 
        {
            io.DefaultImager = new io.SkiaImager(100);
            _dataset = LoadAndPreprocessDataset();
            _lossFunction = nn.BCELoss();
            _batchSize = batchSize;

            _generator = new Generator(generatorInputSize);
            _discriminator = new Discriminator();

            _generatorOptimizer = optim.Adam(_generator.parameters(), lr: learningRate, beta1: optimizerBetas.Item1, beta2: optimizerBetas.Item2);
            _discriminatorOptimizer = optim.Adam(_discriminator.parameters(), lr: learningRate, beta1: optimizerBetas.Item1, beta2: optimizerBetas.Item2);
        }

        internal Tensor LoadAndPreprocessDataset()
        {
            NDArray dataset = np.load(@"Dataset/airplane.npy");
            Tensor ten = from_array(dataset.ToMuliDimArray<byte>()).@float();
            ten = (ten - 127.5) / 127.5; //Normalizacja
            ten = ten.reshape(ten.size(0), 28, 28);
            ten = transforms.Pad(new long[] { 2, 2, 2, 2 }, fill: -1).forward(ten);
            //ten = ten[TensorIndex.Slice(0, 125), TensorIndex.Colon];
            //io.write_png(test.@byte().unsqueeze(0), "testUpsacled.png");
            return ten;
        }
    }
}
