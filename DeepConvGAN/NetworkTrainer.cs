using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torchvision;
using NumSharp;

namespace DeepConvGAN
{
    public class NetworkTrainer
    {
        public NetworkTrainer() 
        {
            io.DefaultImager = new io.SkiaImager(100);
            LoadAndPreprocessDataset();
        }

        internal Tensor LoadAndPreprocessDataset()
        {
            NDArray dataset = np.load(@"Dataset/airplane.npy");
            Tensor ten = from_array(dataset.ToMuliDimArray<byte>()).@float().cuda();
            ten = (ten - 127.5) / 127.5; //Normalizacja
            ten = ten.reshape(ten.size(0), 28, 28);

            //ten = ten[TensorIndex.Slice(0, 125), TensorIndex.Colon];
            return ten;
        }
    }
}
