using TorchSharp;
using static TorchSharp.torchvision;

namespace DeepConvGAN
{
    internal class NetworkTrainer
    {
        public NetworkTrainer() 
        {
            io.DefaultImager = new io.SkiaImager(100);
        }

        internal void LoadAndPreprocessDataset()
        {
            ITransform dataTransform = transforms.Compose(
                transforms.Resize(64, 64), 
                transforms.Grayscale(), 
                transforms.Normalize(new[] {0.5}, new[] {0.5} ));
            

        }
    }
}
