using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using TorchSharp.Modules;
using TorchSharp;

namespace DeepConvGAN
{
    internal class Discriminator : Module<Tensor, Tensor>
    {
        private readonly Sequential _layers;

        internal Discriminator() : base("Discriminator")
        {
            _layers = Sequential(
                Conv2d(1, 32, 4, 2, 1, bias: false),
                LeakyReLU(0.2, inplace: true),

                Conv2d(32, 32 * 2, 4, 2, 1, bias: false),
                BatchNorm2d(32 * 2),
                LeakyReLU(0.2, inplace: true),

                Conv2d(32 * 2, 32 * 4, 4, 2, 1, bias: false),
                BatchNorm2d(32 * 4),
                LeakyReLU(0.2, inplace: true),

                Conv2d(32 * 4, 1, 4, 1, 0, bias: false),
                Sigmoid(),
                Flatten()
            );
            RegisterComponents();
            apply(WeightsInit);
        }

        private void WeightsInit(Module m)
        {
            string name = m.GetName();
            if (name.Contains("Conv"))
                init.normal_(m.parameters().First(), 0.0, 0.02);
            else if (name.Contains("BatchNorm"))
            {
                init.normal_(m.get_parameter("weight"), 1.0, 0.02);
                init.zeros_(m.get_parameter("bias"));
            }
        }

        public override Tensor forward(Tensor x)
        {
            return _layers.forward(x);
        }

        public Tensor CalculateError(Tensor data, Tensor target, WeightedLoss<Tensor, Tensor, Tensor> lossFunction = null)
        {
            lossFunction ??= BCELoss();
            return lossFunction.forward(data, target);
        }
    }
}
