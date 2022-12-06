using static TorchSharp.torch.nn;
using TorchSharp.Modules;
using TorchSharp;

namespace DeepConvGAN
{
    internal class Generator : Module<torch.Tensor, torch.Tensor>
    {
        private readonly Sequential _layers;

        internal Generator() : base("Generator")
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
