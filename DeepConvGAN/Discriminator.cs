using static TorchSharp.torch.nn;
using TorchSharp.Modules;
using TorchSharp;

namespace DeepConvGAN
{
    internal class Discriminator : Module<torch.Tensor, torch.Tensor>
    {
        private readonly Sequential _layers;

        internal Discriminator() : base("Discriminator")
        {
            _layers = Sequential(
                
            );
            RegisterComponents();
        }

        public override torch.Tensor forward(torch.Tensor x)
        {
            return _layers.forward(x);
        }
    }
}
