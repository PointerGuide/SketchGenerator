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
        private readonly int _numEpoch;
        private readonly int _batchSize;
        private readonly int _noiseSize;
        private readonly Device _device;
        private string _runId;
        private const int RealLabel = 1, FakeLabel = 0;
        private const string OutputDirectory = "./runs";

        private readonly Generator _generator;
        private readonly Discriminator _discriminator;

        private readonly TorchSharp.Modules.Adam _generatorOptimizer;
        private readonly TorchSharp.Modules.Adam _discriminatorOptimizer;
        public NetworkTrainer((double, double) optimizerBetas, int numEpoch = 5, int batchSize = 128, 
            int generatorInputSize = 50, double learningRate = 0.0002, DeviceType device = DeviceType.CUDA) 
        {
            io.DefaultImager = new io.SkiaImager(100);
            _device = new Device(device);
            _dataset = LoadAndPreprocessDataset().to(_device);
            _lossFunction = nn.BCELoss().to(_device);
            _numEpoch = numEpoch;
            _batchSize = batchSize;
            _noiseSize = generatorInputSize;
            

            _generator = new Generator(generatorInputSize).to(_device);
            _discriminator = new Discriminator().to(_device);

            _generatorOptimizer = optim.Adam(_generator.parameters(), lr: learningRate, beta1: optimizerBetas.Item1, beta2: optimizerBetas.Item2);
            _discriminatorOptimizer = optim.Adam(_discriminator.parameters(), lr: learningRate, beta1: optimizerBetas.Item1, beta2: optimizerBetas.Item2);
        }

        internal Tensor LoadAndPreprocessDataset()
        {
            NDArray dataset = np.load(@"Dataset/bicycle.npy");
            Tensor ten = from_array(dataset.ToMuliDimArray<byte>()).@float();
            ten = (ten - 127.5) / 127.5; //Normalizacja
            ten = ten.reshape(ten.size(0), 1, 28, 28);
            ten = transforms.Pad(new long[] { 2, 2, 2, 2 }, fill: -1).forward(ten);
            return ten;
        }

        private void PrepareUniqueRun()
        {
            Console.WriteLine($"Preparing training run with ID: {_runId}");
            if (!Directory.Exists(OutputDirectory))
                Directory.CreateDirectory(OutputDirectory);
            Directory.CreateDirectory($@"{OutputDirectory}/{_runId}");
        }

        private Tensor GenerateNoise(int batchSize = 128) => rand(batchSize, _noiseSize, 1, 1, device: _device);
        public void Train()
        {
            _runId = DateTime.Now.ToString().Replace(':', '.');
            PrepareUniqueRun();
            for (int epoch = 1; epoch <= _numEpoch; epoch++)
            {
                string epochDirectory = $@"{OutputDirectory}/{_runId}/epoch{epoch}";
                if (!Directory.Exists(epochDirectory))
                    Directory.CreateDirectory(epochDirectory);
                int dataIdx = 0;
                int minibatchNo = 0;
                while (dataIdx < _dataset.size(0))
                {
                    _discriminatorOptimizer.zero_grad();
                    using (torch.NewDisposeScope())
                    {
                        //1. Prepare data and labels
                        Tensor data = _dataset[TensorIndex.Slice(dataIdx, dataIdx + _batchSize), TensorIndex.Colon];

                        Tensor realTarget = ones(data.size(0)).unsqueeze(1).to(_device);
                        Tensor fakeTarget = zeros(data.size(0)).unsqueeze(1).to(_device);


                        //2. Train discriminator with real labels
                        Tensor discriminatorRealLoss = _discriminator.CalculateError(_discriminator.forward(data), realTarget);
                        discriminatorRealLoss.backward();

                        Tensor noise = GenerateNoise((int)data.size(0)).to(_device);
                        Tensor generatedImage = _generator.forward(noise);
                        Tensor output = _discriminator.forward(generatedImage.detach());

                        //3. Train discriminator with fake labels
                        Tensor discriminatorFakeLoss = _discriminator.CalculateError(output, fakeTarget);
                        discriminatorFakeLoss.backward();

                        double discriminatorTotalLoss = discriminatorRealLoss.ToSingle() + discriminatorFakeLoss.ToSingle();
                        _discriminatorOptimizer.step();


                        //4. Train generator with real labels
                        _generatorOptimizer.zero_grad();
                        Tensor generatorLoss = _generator.CalculateError(_discriminator.forward(generatedImage), realTarget);
                        double generatorErr = generatorLoss.ToSingle();
                        generatorLoss.backward();
                        _generatorOptimizer.step();

                        if (minibatchNo % 50 == 0)
                        {
                            Console.WriteLine($"Epoch {epoch}: Generator loss {generatorErr}, Discriminator loss {discriminatorTotalLoss}");
                            Tensor img = generatedImage[0]*127.5+127.5;
                            io.write_png(img.@byte().cpu(), $"{epochDirectory}/image-minibatch{minibatchNo}.png");
                        }

                        minibatchNo++;
                    }

                    dataIdx += _batchSize;
                }
                _generator.Save($"{epochDirectory}/generator");
                _discriminator.Save($"{epochDirectory}/discriminator");
            }
            
        }
    }
}
