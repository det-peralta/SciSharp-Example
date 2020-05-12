namespace predictor
{
    class Program
    {

        static void Main(string[] args)
        {
            Helpers.PrepareData();
            Helpers.Predict();
        }
    }
}
