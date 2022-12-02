using TorchSharp.Modules;
using TorchSharp;
using static TorchSharp.torch.nn;

namespace SketchGenerator;

internal class Generator : Module<torch.Tensor, torch.Tensor>
{
    private readonly Sequential _layers;

    internal Generator(long noiseDimensions, long generatorOutputSize) : base("Generator")
    { 
        _layers = Sequential(
            //First up-sampling layers
            Linear(noiseDimensions, 128, hasBias: false),
            BatchNorm1d(128, 0.8),
            LeakyReLU(0.25),
            //Second up-sampling layers
            Linear(128, 256, hasBias: false),
            BatchNorm1d(256, 0.8),
            LeakyReLU(0.25),
            //Third up-sampling layers
            Linear(256, 512, hasBias: false),
            BatchNorm1d(512, 0.8),
            LeakyReLU(0.25),
            //Final up-sampling layers
            Linear(512, generatorOutputSize, hasBias: false),
            Tanh()
        );
        RegisterComponents();
    }

    public override torch.Tensor forward(torch.Tensor x)
    {
        return _layers.forward(x);
    }
}