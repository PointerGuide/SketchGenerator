using static TorchSharp.torch.nn;
using TorchSharp;
using TorchSharp.Modules;

namespace SketchGenerator;

internal class Discriminator : Module<torch.Tensor, torch.Tensor>
{
    private readonly Sequential _layers;
    internal Discriminator(long discriminatorInputSize) : base("Discriminator")
    {
        _layers = Sequential(
            Linear(discriminatorInputSize, 1024),
            LeakyReLU(0.25),
            Linear(1024, 512),
            LeakyReLU(0.25),
            Linear(512, 256),
            LeakyReLU(0.25),
            Linear(256, 1),
            Sigmoid()
        );
        RegisterComponents();
    }

    public override torch.Tensor forward(torch.Tensor x)
    {
        return _layers.forward(x);
    }
}