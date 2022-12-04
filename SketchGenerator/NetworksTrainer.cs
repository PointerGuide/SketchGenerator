using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch.utils.data;
using static TorchSharp.torchvision.datasets;

namespace SketchGenerator;

internal class NetworksTrainer
{
    private readonly DeviceType _deviceType;
    private readonly Generator _generator;
    private readonly Discriminator _discriminator;
    private readonly long _noiseDimensions;
    private readonly double _optimizersLR;
    private readonly (double, double) _optimizersBetas;
    private readonly string _runId;
    private readonly BCELoss _lossFunction;
    private readonly AdamW _generatorOptimizer;
    private readonly AdamW _discriminatorOptimizer;
    private const string OutputDirectory = "./runs";

    public NetworksTrainer(DeviceType deviceType, long noiseDimensions, long imageOutputSize, 
        double optimizersLr, (double, double) optimizersBetas)
    {
        _deviceType = deviceType;
        _runId = DateTime.Now.ToString().Replace(':', '.');
        _generator = new Generator(noiseDimensions, imageOutputSize).to(_deviceType);
        _discriminator = new Discriminator(imageOutputSize).to(_deviceType);
        _noiseDimensions = noiseDimensions;
        _optimizersLR = optimizersLr;
        _optimizersBetas = optimizersBetas;

        _lossFunction = new BCELoss();
        _generatorOptimizer = torch.optim.AdamW(_generator.parameters(),
            lr: _optimizersLR, beta1: _optimizersBetas.Item1, beta2: _optimizersBetas.Item2);
        _discriminatorOptimizer = torch.optim.AdamW(_discriminator.parameters(),
            lr: optimizersLr, beta1: _optimizersBetas.Item1, beta2: _optimizersBetas.Item2);
    }
    private void PrepareUniqueRun()
    {
        Console.WriteLine($"Preparing training run with ID: {_runId}");
        if (!Directory.Exists(OutputDirectory))
            Directory.CreateDirectory(OutputDirectory);
        Directory.CreateDirectory($@"{OutputDirectory}/{_runId}");
    }

    private void SaveModels(int epoch)
    {
        _generator.Save($@"{OutputDirectory}/{_runId}/generator_{epoch}.pth");
        _discriminator.Save($@"{OutputDirectory}/{_runId}/discriminator_{epoch}.pth");
    }

    public void Train(int numEpoch, int batchSize)
    {
        PrepareUniqueRun();
        DataLoader dataLoader = PrepareTrainSet(batchSize);
        
        for (int epoch = 1; epoch <= numEpoch; epoch++)
        {
            Console.WriteLine($"Performing epoch: {epoch}...");
            using (torch.NewDisposeScope())
            {
                uint batchNum = 1;
                foreach (Dictionary<string, torch.Tensor>? batch in dataLoader)
                {
                    (float generatorLossValue, float discriminatorLossValue) = PerformTrainStep(batch);
                    if (batchNum % 50 == 0)
                    {
                        Console.WriteLine(
                            $"Losses after mini-batch {batchNum}: generator {generatorLossValue} discriminator {discriminatorLossValue}");
                        GenerateImage(_generator, epoch, batchNum, batchSize);
                    }
                    batchNum++;
                }
            }
            //SaveModels(epoch);
        }
        Console.WriteLine($"Finished run: {_runId}");
    }

    private void GenerateImage(Generator generator, int epoch, uint batchNum, int batchSize)
    {
        torch.Tensor noise = torch.randn(batchSize, _noiseDimensions, device: new torch.Device(_deviceType));
        generator.eval();
        torch.Tensor images = generator.forward(noise);
        for (int i = 0; i < 16; i++)
        {
            torch.Tensor image = images[i];
            image = image.cpu().detach().reshape(28, 28).unsqueeze(0);
            torchvision.io.write_image(image.@byte(), "test.png", torchvision.ImageFormat.Png,
                new torchvision.io.SkiaImager(100));
        }
    }

    private (float errorGenerator, float errorDiscriminator) PerformTrainStep(Dictionary<string, torch.Tensor> batch)
    {
        //1. Preparation - set real and fake labels//
        double realLabelValue = 1.0, fakeLabelValue = 0.0;
        torch.Tensor realImages = batch["data"];
        torch.Tensor realLabels = batch["label"];
        long dataSize = realImages.size(0);
        torch.Tensor label = torch.full(size: new[] { dataSize, 1 }, realLabelValue, device: new torch.Device(_deviceType));


        //2. Training the discriminator//
        //Forward + backward on real images
        realImages = realImages.view(realImages.size(0), -1);
        float errorRealImages = CalculateForwardAndBackwardError(_discriminator, realImages, label, _lossFunction);
        //Forward + backward on generated images
        torch.Tensor noise = torch.randn(dataSize, _noiseDimensions, device: new torch.Device(_deviceType));
        torch.Tensor generatedImages = _generator.forward(noise);
        label.fill_(fakeLabelValue);
        float errorGeneratedImages = CalculateForwardAndBackwardError(_discriminator, generatedImages.detach(), label, _lossFunction);
        _discriminatorOptimizer.step();


        //3. Training the generator//
        label.fill_(realLabelValue);
        float errorGenerator = CalculateForwardAndBackwardError(_discriminator, generatedImages, label, _lossFunction);
        _generatorOptimizer.step();


        //4. Compute results//
        //Compute loss values in floats for discriminator, which is joint loss.
        float errorDiscriminator = errorRealImages + errorGeneratedImages;
        return (errorGenerator, errorDiscriminator);
    }

    private float CalculateForwardAndBackwardError(torch.nn.Module model, torch.Tensor data, torch.Tensor targets,
        WeightedLoss<torch.Tensor, torch.Tensor, torch.Tensor> lossFunction)
    {
        torch.Tensor outputs = _discriminator.forward(data);
        torch.Tensor error = _lossFunction.forward(outputs, targets);
        error.backward();
        return error.to(DeviceType.CPU).item<float>();
    }

    private DataLoader PrepareTrainSet(int batchSize)
    {
        Dataset dataset = MNIST(Directory.GetCurrentDirectory(), train: true, download: true,
            target_transform: torchvision.transforms.Normalize(new[] { 0.5 }, new[] { 0.5 }));
        return new DataLoader(dataset, batchSize: batchSize, shuffle: true,
            num_worker: 12, device: new torch.Device(_deviceType));

    }
}