using System;

namespace ConsoleApplication
{
    public class Program
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            int numFeatures = 9;
            int numRows = 300;
            int seed = 42;  // interesting seeds: 28, 32, (42), 56, 58, 63, 91

            // generate artificial observations
            Console.WriteLine("\nGenerating " + numRows +
            " artificial data items with " + numFeatures + " features");    
            //double[][] allData = MakeAllData(numFeatures, numRows, seed);
            double[][] allData = InputData(numFeatures, numRows, seed);
            // split into training and test datasets
            Console.WriteLine("Creating train (80%) and test (20%) matrices after scrambling observations order..");
            double[][] trainData;
            double[][] testData;
            MakeTrainTest(allData, 0, out trainData, out testData);
            Console.WriteLine("Done");
            Console.WriteLine("\nTraining data: \n");
            ShowMatrix(trainData, 4, 2, true);
            Console.WriteLine("\nTest data: \n");
            ShowMatrix(testData, 3, 2, true);        
            
            // instantiate logistic binary classifier
            Console.WriteLine("Creating LR binary classifier..");
            LogisticClassifier lc = new LogisticClassifier(numFeatures);


            // find L1 and L2
            Console.WriteLine("\nSeeking good L1 weight");
            double alpha1 = lc.FindGoodL1Weight(trainData, seed);
            Console.WriteLine("Good L1 weight = " + alpha1.ToString("F3"));

            Console.WriteLine("\nSeeking good L2 weight");
            double alpha2 = lc.FindGoodL2Weight(trainData, seed);
            Console.WriteLine("Good L2 weight = " + alpha2.ToString("F3"));

            // train using no regularization
            int maxEpochs = 1000;
            Console.WriteLine("\nStarting training using no regularization..");
            double[] weights = lc.TrainWeights(trainData, maxEpochs, seed, 0.0, 0.0);

            Console.WriteLine("\nBest weights found:");
            ShowVector(weights, 3, weights.Length, true);

            double trainAccuracy = lc.Accuracy(trainData, weights);
            Console.WriteLine("Prediction accuracy on training data = " + trainAccuracy.ToString("F4"));

            double testAccuracy = lc.Accuracy(testData, weights);
            Console.WriteLine("Prediction accuracy on test data = " + testAccuracy.ToString("F4"));
            // train using L1 regularization
            Console.WriteLine("\nStarting training using L1 regularization, alpha1 = " + alpha1.ToString("F3"));
            weights = lc.TrainWeights(trainData, maxEpochs, seed, alpha1, 0.0);

            Console.WriteLine("\nBest weights found:");
            ShowVector(weights, 3, weights.Length, true);

            trainAccuracy = lc.Accuracy(trainData, weights);
            Console.WriteLine("Prediction accuracy on training data = " + trainAccuracy.ToString("F4"));

            testAccuracy = lc.Accuracy(testData, weights);
            Console.WriteLine("Prediction accuracy on test data = " + testAccuracy.ToString("F4"));

            // train using L2 regularization
            Console.WriteLine("\nStarting training using L2 regularization, alpha2 = " + alpha2.ToString("F3"));
            weights = lc.TrainWeights(trainData, maxEpochs, seed, 0.0, alpha2);

            Console.WriteLine("\nBest weights found:");
            ShowVector(weights, 3, weights.Length, true);

            trainAccuracy = lc.Accuracy(trainData, weights);
            Console.WriteLine("Prediction accuracy on training data = " + trainAccuracy.ToString("F4"));

            testAccuracy = lc.Accuracy(testData, weights);
            Console.WriteLine("Prediction accuracy on test data = " + testAccuracy.ToString("F4"));

            Console.WriteLine("\nEnd Regularization demo\n");
            Console.ReadLine();

        }

        private static double[][] InputData(int numFeatures, int numRows, int seed)
        {
            Random rnd = new Random(seed);

                double[] weights = new double[numFeatures]; // inc. b0
                for (int i = 0; i < weights.Length; ++i)
                    weights[i] = 2.0 * rnd.NextDouble() - 1.0;

                double[][] allData = new double[300][];
                //allData = new double[1000][];
                allData[0] = new double[] { 1, 160, 12.00, 5.73, 23.11, 1, 49, 25.30, 97.20, 52, 1 };
                allData[1] = new double[] { 2, 144, 0.01, 4.41, 28.61, 0, 55, 28.87, 2.06, 63, 1 };
                allData[2] = new double[] { 3, 118, 0.08, 3.48, 32.28, 1, 52, 29.14, 3.81, 46, 0 };
                allData[3] = new double[] { 4, 170, 7.50, 6.41, 38.03, 1, 51, 31.99, 24.26, 58, 1 };
                allData[4] = new double[] { 5, 134, 13.60, 3.50, 27.78, 1, 60, 25.99, 57.34, 49, 1 };
                allData[5] = new double[] { 6, 132, 6.20, 6.47, 36.21, 1, 62, 30.77, 14.14, 45, 0 };
                allData[6] = new double[] { 7, 142, 4.05, 3.38, 16.20, 0, 59, 20.81, 2.62, 38, 0 };
                allData[7] = new double[] { 8, 114, 4.08, 4.59, 14.60, 1, 62, 23.11, 6.72, 58, 1 };
                allData[8] = new double[] { 9, 114, 0.00, 3.83, 19.40, 1, 49, 24.86, 2.49, 29, 0 };
                allData[9] = new double[] { 10, 132, 0.00, 5.80, 30.96, 1, 69, 30.11, 0.00, 53, 1 };
                allData[10] = new double[] { 11, 206, 6.00, 2.95, 32.27, 0, 72, 26.81, 56.06, 60, 1 };
                allData[11] = new double[] { 12, 134, 14.10, 4.44, 22.39, 1, 65, 23.09, 0.00, 40, 1 };
                allData[12] = new double[] { 13, 118, 0.00, 1.88, 10.05, 0, 59, 21.57, 0.00, 17, 0 };
                allData[13] = new double[] { 14, 132, 0.00, 1.87, 17.21, 0, 49, 23.63, 0.97, 15, 0 };
                allData[14] = new double[] { 15, 112, 9.65, 2.29, 17.20, 1, 54, 23.53, 0.68, 53, 0 };
                allData[15] = new double[] { 16, 117, 1.53, 2.44, 28.95, 1, 35, 25.89, 30.03, 46, 0 };
                allData[16] = new double[] { 17, 120, 7.50, 15.33, 22.00, 0, 60, 25.31, 34.49, 49, 0 };
                allData[17] = new double[] { 18, 146, 10.50, 8.29, 35.36, 1, 78, 32.73, 13.89, 53, 1 };
                allData[18] = new double[] { 19, 158, 2.60, 7.46, 34.07, 1, 61, 29.30, 53.28, 62, 1 };
                allData[19] = new double[] { 20, 124, 14.00, 6.23, 35.96, 1, 45, 30.09, 0.00, 59, 1 };
                allData[20] = new double[] { 21, 106, 1.61, 1.74, 12.32, 0, 74, 20.92, 13.37, 20, 1 };
                allData[21] = new double[] { 22, 132, 7.90, 2.85, 26.50, 1, 51, 26.16, 25.71, 44, 0 };
                allData[22] = new double[] { 23, 150, 0.30, 6.38, 33.99, 1, 62, 24.64, 0.00, 50, 0 };
                allData[23] = new double[] { 24, 138, 0.60, 3.81, 28.66, 0, 54, 28.70, 1.46, 58, 0 };
                allData[24] = new double[] { 25, 142, 18.20, 4.34, 24.38, 0, 61, 26.19, 0.00, 50, 0 };
                allData[25] = new double[] { 26, 124, 4.00, 12.42, 31.29, 1, 54, 23.23, 2.06, 42, 1 };
                allData[26] = new double[] { 27, 118, 6.00, 9.65, 33.91, 0, 60, 38.80, 0.00, 48, 0 };
                allData[27] = new double[] { 28, 145, 9.10, 5.24, 27.55, 0, 59, 20.96, 21.60, 61, 1 };
                allData[28] = new double[] { 29, 144, 4.09, 5.55, 31.40, 1, 60, 29.43, 5.55, 56, 0 };
                allData[29] = new double[] { 30, 146, 0.00, 6.62, 25.69, 0, 60, 28.07, 8.23, 63, 1 };
                allData[30] = new double[] { 31, 136, 2.52, 3.95, 25.63, 0, 51, 21.86, 0.00, 45, 1 };
                allData[31] = new double[] { 32, 158, 1.02, 6.33, 23.88, 0, 66, 22.13, 24.99, 46, 1 };
                allData[32] = new double[] { 33, 122, 6.60, 5.58, 35.95, 1, 53, 28.07, 12.55, 59, 1 };
                allData[33] = new double[] { 34, 126, 8.75, 6.53, 34.02, 0, 49, 30.25, 0.00, 41, 1 };
                allData[34] = new double[] { 35, 148, 5.50, 7.10, 25.31, 0, 56, 29.84, 3.60, 48, 0 };
                allData[35] = new double[] { 36, 122, 4.26, 4.44, 13.04, 0, 57, 19.49, 48.99, 28, 1 };
                allData[36] = new double[] { 37, 140, 3.90, 7.32, 25.05, 0, 47, 27.36, 36.77, 32, 0 };
                allData[37] = new double[] { 38, 110, 4.64, 4.55, 30.46, 0, 48, 30.90, 15.22, 46, 0 };
                allData[38] = new double[] { 39, 130, 0.00, 2.82, 19.63, 1, 70, 24.86, 0.00, 29, 0 };
                allData[39] = new double[] { 40, 136, 11.20, 5.81, 31.85, 1, 75, 27.68, 22.94, 58, 1 };
                allData[40] = new double[] { 41, 118, 0.28, 5.80, 33.70, 1, 60, 30.98, 0.00, 41, 1 };
                allData[41] = new double[] { 42, 144, 0.04, 3.38, 23.61, 0, 30, 23.75, 4.66, 30, 0 };
                allData[42] = new double[] { 43, 120, 0.00, 1.07, 16.02, 0, 47, 22.15, 0.00, 15, 0 };
                allData[43] = new double[] { 44, 130, 2.61, 2.72, 22.99, 1, 51, 26.29, 13.37, 51, 1 };
                allData[44] = new double[] { 45, 114, 0.00, 2.99, 9.74, 0, 54, 46.58, 0.00, 17, 0 };
                allData[45] = new double[] { 46, 128, 4.65, 3.31, 22.74, 0, 62, 22.95, 0.51, 48, 0 };
                allData[46] = new double[] { 47, 162, 7.40, 8.55, 24.65, 1, 64, 25.71, 5.86, 58, 1 };
                allData[47] = new double[] { 48, 116, 1.91, 7.56, 26.45, 1, 52, 30.01, 3.60, 33, 1 };
                allData[48] = new double[] { 49, 114, 0.00, 1.94, 11.02, 0, 54, 20.17, 38.98, 16, 0 };
                allData[49] = new double[] { 50, 126, 3.80, 3.88, 31.79, 0, 57, 30.53, 0.00, 30, 0 };
                allData[50] = new double[] { 51, 122, 0.00, 5.75, 30.90, 1, 46, 29.01, 4.11, 42, 0 };
                allData[51] = new double[] { 52, 134, 2.50, 3.66, 30.90, 0, 52, 27.19, 23.66, 49, 0 };
                allData[52] = new double[] { 53, 152, 0.90, 9.12, 30.23, 0, 56, 28.64, 0.37, 42, 1 };
                allData[53] = new double[] { 54, 134, 8.08, 1.55, 17.50, 1, 56, 22.65, 66.65, 31, 1 };
                allData[54] = new double[] { 55, 156, 3.00, 1.82, 27.55, 0, 60, 23.91, 54.00, 53, 0 };
                allData[55] = new double[] { 56, 152, 5.99, 7.99, 32.48, 0, 45, 26.57, 100.32, 48, 0 };
                allData[56] = new double[] { 57, 118, 0.00, 2.99, 16.17, 0, 49, 23.83, 3.22, 28, 0 };
                allData[57] = new double[] { 58, 126, 5.10, 2.96, 26.50, 0, 55, 25.52, 12.34, 38, 1 };
                allData[58] = new double[] { 59, 103, 0.03, 4.21, 18.96, 0, 48, 22.94, 2.62, 18, 0 };
                allData[59] = new double[] { 60, 121, 0.80, 5.29, 18.95, 1, 47, 22.51, 0.00, 61, 0 };
                allData[60] = new double[] { 61, 142, 0.28, 1.80, 21.03, 0, 57, 23.65, 2.93, 33, 0 };
                allData[61] = new double[] { 62, 138, 1.15, 5.09, 27.87, 1, 61, 25.65, 2.34, 44, 0 };
                allData[62] = new double[] { 63, 152, 10.10, 4.71, 24.65, 1, 65, 26.21, 24.53, 57, 0 };
                allData[63] = new double[] { 64, 140, 0.45, 4.30, 24.33, 0, 41, 27.23, 10.08, 38, 0 };
                allData[64] = new double[] { 65, 130, 0.00, 1.82, 10.45, 0, 57, 22.07, 2.06, 17, 0 };
                allData[65] = new double[] { 66, 136, 7.36, 2.19, 28.11, 1, 61, 25.00, 61.71, 54, 0 };
                allData[66] = new double[] { 67, 124, 4.82, 3.24, 21.10, 1, 48, 28.49, 8.42, 30, 0 };
                allData[67] = new double[] { 68, 112, 0.41, 1.88, 10.29, 0, 39, 22.08, 20.98, 27, 0 };
                allData[68] = new double[] { 69, 118, 4.46, 7.27, 29.13, 1, 48, 29.01, 11.11, 33, 0 };
                allData[69] = new double[] { 70, 122, 0.00, 3.37, 16.10, 0, 67, 21.06, 0.00, 32, 1 };
                allData[70] = new double[] { 71, 118, 0.00, 3.67, 12.13, 0, 51, 19.15, 0.60, 15, 0 };
                allData[71] = new double[] { 72, 130, 1.72, 2.66, 10.38, 0, 68, 17.81, 11.10, 26, 0 };
                allData[72] = new double[] { 73, 130, 5.60, 3.37, 24.80, 0, 58, 25.76, 43.20, 36, 0 };
                allData[73] = new double[] { 74, 126, 0.09, 5.03, 13.27, 1, 50, 17.75, 4.63, 20, 0 };
                allData[74] = new double[] { 75, 128, 0.40, 6.17, 26.35, 0, 64, 27.86, 11.11, 34, 0 };
                allData[75] = new double[] { 76, 136, 0.00, 4.12, 17.42, 0, 52, 21.66, 12.86, 40, 0 };
                allData[76] = new double[] { 77, 134, 0.00, 5.90, 30.84, 0, 49, 29.16, 0.00, 55, 0 };
                allData[77] = new double[] { 78, 140, 0.60, 5.56, 33.39, 1, 58, 27.19, 0.00, 55, 1 };
                allData[78] = new double[] { 79, 168, 4.50, 6.68, 28.47, 0, 43, 24.25, 24.38, 56, 1 };
                allData[79] = new double[] { 80, 108, 0.40, 5.91, 22.92, 1, 57, 25.72, 72.00, 39, 0 };
                allData[80] = new double[] { 81, 114, 3.00, 7.04, 22.64, 1, 55, 22.59, 0.00, 45, 1 };
                allData[81] = new double[] { 82, 140, 8.14, 4.93, 42.49, 0, 53, 45.72, 6.43, 53, 1 };
                allData[82] = new double[] { 83, 148, 4.80, 6.09, 36.55, 1, 63, 25.44, 0.88, 55, 1 };
                allData[83] = new double[] { 84, 148, 12.20, 3.79, 34.15, 0, 57, 26.38, 14.40, 57, 1 };
                allData[84] = new double[] { 85, 128, 0.00, 2.43, 13.15, 1, 63, 20.75, 0.00, 17, 0 };
                allData[85] = new double[] { 86, 130, 0.56, 3.30, 30.86, 0, 49, 27.52, 33.33, 45, 0 };
                allData[86] = new double[] { 87, 126, 10.50, 4.49, 17.33, 0, 67, 19.37, 0.00, 49, 1 };
                allData[87] = new double[] { 88, 140, 0.00, 5.08, 27.33, 1, 41, 27.83, 1.25, 38, 0 };
                allData[88] = new double[] { 89, 126, 0.90, 5.64, 17.78, 1, 55, 21.94, 0.00, 41, 0 };
                allData[89] = new double[] { 90, 122, 0.72, 4.04, 32.38, 0, 34, 28.34, 0.00, 55, 0 };
                allData[90] = new double[] { 91, 116, 1.03, 2.83, 10.85, 0, 45, 21.59, 1.75, 21, 0 };
                allData[91] = new double[] { 92, 120, 3.70, 4.02, 39.66, 0, 61, 30.57, 0.00, 64, 1 };
                allData[92] = new double[] { 93, 143, 0.46, 2.40, 22.87, 0, 62, 29.17, 15.43, 29, 0 };
                allData[93] = new double[] { 94, 118, 4.00, 3.95, 18.96, 0, 54, 25.15, 8.33, 49, 1 };
                allData[94] = new double[] { 95, 194, 1.70, 6.32, 33.67, 0, 47, 30.16, 0.19, 56, 0 };
                allData[95] = new double[] { 96, 134, 3.00, 4.37, 23.07, 0, 56, 20.54, 9.65, 62, 0 };
                allData[96] = new double[] { 97, 138, 2.16, 4.90, 24.83, 1, 39, 26.06, 28.29, 29, 0 };
                allData[97] = new double[] { 98, 136, 0.00, 5.00, 27.58, 1, 49, 27.59, 1.47, 39, 0 };
                allData[98] = new double[] { 99, 122, 3.20, 11.32, 35.36, 1, 55, 27.07, 0.00, 51, 1 };
                allData[99] = new double[] { 100, 164, 12.00, 3.91, 19.59, 0, 51, 23.44, 19.75, 39, 0 };
                allData[100] = new double[] { 101, 136, 8.00, 7.85, 23.81, 1, 51, 22.69, 2.78, 50, 0 };
                allData[101] = new double[] { 102, 166, 0.07, 4.03, 29.29, 0, 53, 28.37, 0.00, 27, 0 };
                allData[102] = new double[] { 103, 118, 0.00, 4.34, 30.12, 1, 52, 32.18, 3.91, 46, 0 };
                allData[103] = new double[] { 104, 128, 0.42, 4.60, 26.68, 0, 41, 30.97, 10.33, 31, 0 };
                allData[104] = new double[] { 105, 118, 1.50, 5.38, 25.84, 0, 64, 28.63, 3.89, 29, 0 };
                allData[105] = new double[] { 106, 158, 3.60, 2.97, 30.11, 0, 63, 26.64, 108.00, 64, 0 };
                allData[106] = new double[] { 107, 108, 1.50, 4.33, 24.99, 0, 66, 22.29, 21.60, 61, 1 };
                allData[107] = new double[] { 108, 170, 7.60, 5.50, 37.83, 1, 42, 37.41, 6.17, 54, 1 };
                allData[108] = new double[] { 109, 118, 1.00, 5.76, 22.10, 0, 62, 23.48, 7.71, 42, 0 };
                allData[109] = new double[] { 110, 124, 0.00, 3.04, 17.33, 0, 49, 22.04, 0.00, 18, 0 };
                allData[110] = new double[] { 111, 114, 0.00, 8.01, 21.64, 0, 66, 25.51, 2.49, 16, 0 };
                allData[111] = new double[] { 112, 168, 9.00, 8.53, 24.48, 1, 69, 26.18, 4.63, 54, 1 };
                allData[112] = new double[] { 113, 134, 2.00, 3.66, 14.69, 0, 52, 21.03, 2.06, 37, 0 };
                allData[113] = new double[] { 114, 174, 0.00, 8.46, 35.10, 1, 35, 25.27, 0.00, 61, 1 };
                allData[114] = new double[] { 115, 116, 31.20, 3.17, 14.99, 0, 47, 19.40, 49.06, 59, 1 };
                allData[115] = new double[] { 116, 128, 0.00, 10.58, 31.81, 1, 46, 28.41, 14.66, 48, 0 };
                allData[116] = new double[] { 117, 140, 4.50, 4.59, 18.01, 0, 63, 21.91, 22.09, 32, 1 };
                allData[117] = new double[] { 118, 154, 0.70, 5.91, 25.00, 0, 13, 20.60, 0.00, 42, 0 };
                allData[118] = new double[] { 119, 150, 3.50, 6.99, 25.39, 1, 50, 23.35, 23.48, 61, 1 };
                allData[119] = new double[] { 120, 130, 0.00, 3.92, 25.55, 0, 68, 28.02, 0.68, 27, 0 };
                allData[120] = new double[] { 121, 128, 2.00, 6.13, 21.31, 0, 66, 22.86, 11.83, 60, 0 };
                allData[121] = new double[] { 122, 120, 1.40, 6.25, 20.47, 0, 60, 25.85, 8.51, 28, 0 };
                allData[122] = new double[] { 123, 120, 0.00, 5.01, 26.13, 0, 64, 26.21, 12.24, 33, 0 };
                allData[123] = new double[] { 124, 138, 4.50, 2.85, 30.11, 0, 55, 24.78, 24.89, 56, 1 };
                allData[124] = new double[] { 125, 153, 7.80, 3.96, 25.73, 0, 54, 25.91, 27.03, 45, 0 };
                allData[125] = new double[] { 126, 123, 8.60, 11.17, 35.28, 1, 70, 33.14, 0.00, 59, 1 };
                allData[126] = new double[] { 127, 148, 4.04, 3.99, 20.69, 0, 60, 27.78, 1.75, 28, 0 };
                allData[127] = new double[] { 128, 136, 3.96, 2.76, 30.28, 1, 50, 34.42, 18.51, 38, 0 };
                allData[128] = new double[] { 129, 134, 8.80, 7.41, 26.84, 0, 35, 29.44, 29.52, 60, 1 };
                allData[129] = new double[] { 130, 152, 12.18, 4.04, 37.83, 1, 63, 34.57, 4.17, 64, 0 };
                allData[130] = new double[] { 131, 158, 13.50, 5.04, 30.79, 0, 54, 24.79, 21.50, 62, 0 };
                allData[131] = new double[] { 132, 132, 2.00, 3.08, 35.39, 0, 45, 31.44, 79.82, 58, 1 };
                allData[132] = new double[] { 133, 134, 1.50, 3.73, 21.53, 0, 41, 24.70, 11.11, 30, 1 };
                allData[133] = new double[] { 134, 142, 7.44, 5.52, 33.97, 0, 47, 29.29, 24.27, 54, 0 };
                allData[134] = new double[] { 135, 134, 6.00, 3.30, 28.45, 0, 65, 26.09, 58.11, 40, 0 };
                allData[135] = new double[] { 136, 122, 4.18, 9.05, 29.27, 1, 44, 24.05, 19.34, 52, 1 };
                allData[136] = new double[] { 137, 116, 2.70, 3.69, 13.52, 0, 55, 21.13, 18.51, 32, 0 };
                allData[137] = new double[] { 138, 128, 0.50, 3.70, 12.81, 1, 66, 21.25, 22.73, 28, 0 };
                allData[138] = new double[] { 139, 120, 0.00, 3.68, 12.24, 0, 51, 20.52, 0.51, 20, 0 };
                allData[139] = new double[] { 140, 124, 0.00, 3.95, 36.35, 1, 59, 32.83, 9.59, 54, 0 };
                allData[140] = new double[] { 141, 160, 14.00, 5.90, 37.12, 0, 58, 33.87, 3.52, 54, 1 };
                allData[141] = new double[] { 142, 130, 2.78, 4.89, 9.39, 1, 63, 19.30, 17.47, 25, 1 };
                allData[142] = new double[] { 143, 128, 2.80, 5.53, 14.29, 0, 64, 24.97, 0.51, 38, 0 };
                allData[143] = new double[] { 144, 130, 4.50, 5.86, 37.43, 0, 61, 31.21, 32.30, 58, 0 };
                allData[144] = new double[] { 145, 109, 1.20, 6.14, 29.26, 0, 47, 24.72, 10.46, 40, 0 };
                allData[145] = new double[] { 146, 144, 0.00, 3.84, 18.72, 0, 56, 22.10, 4.80, 40, 0 };
                allData[146] = new double[] { 147, 118, 1.05, 3.16, 12.98, 1, 46, 22.09, 16.35, 31, 0 };
                allData[147] = new double[] { 148, 136, 3.46, 6.38, 32.25, 1, 43, 28.73, 3.13, 43, 1 };
                allData[148] = new double[] { 149, 136, 1.50, 6.06, 26.54, 0, 54, 29.38, 14.50, 33, 1 };
                allData[149] = new double[] { 150, 124, 15.50, 5.05, 24.06, 0, 46, 23.22, 0.00, 61, 1 };
                allData[150] = new double[] { 151, 148, 6.00, 6.49, 26.47, 0, 48, 24.70, 0.00, 55, 0 };
                allData[151] = new double[] { 152, 128, 6.60, 3.58, 20.71, 0, 55, 24.15, 0.00, 52, 0 };
                allData[152] = new double[] { 153, 122, 0.28, 4.19, 19.97, 0, 61, 25.63, 0.00, 24, 0 };
                allData[153] = new double[] { 154, 108, 0.00, 2.74, 11.17, 0, 53, 22.61, 0.95, 20, 0 };
                allData[154] = new double[] { 155, 124, 3.04, 4.80, 19.52, 1, 60, 21.78, 147.19, 41, 1 };
                allData[155] = new double[] { 156, 138, 8.80, 3.12, 22.41, 1, 63, 23.33, 120.03, 55, 1 };
                allData[156] = new double[] { 157, 127, 0.00, 2.81, 15.70, 0, 42, 22.03, 1.03, 17, 0 };
                allData[157] = new double[] { 158, 174, 9.45, 5.13, 35.54, 0, 55, 30.71, 59.79, 53, 0 };
                allData[158] = new double[] { 159, 122, 0.00, 3.05, 23.51, 0, 46, 25.81, 0.00, 38, 0 };
                allData[159] = new double[] { 160, 144, 6.75, 5.45, 29.81, 0, 53, 25.62, 26.23, 43, 1 };
                allData[160] = new double[] { 161, 126, 1.80, 6.22, 19.71, 0, 65, 24.81, 0.69, 31, 0 };
                allData[161] = new double[] { 162, 208, 27.40, 3.12, 26.63, 0, 66, 27.45, 33.07, 62, 1 };
                allData[162] = new double[] { 163, 138, 0.00, 2.68, 17.04, 0, 42, 22.16, 0.00, 16, 0 };
                allData[163] = new double[] { 164, 148, 0.00, 3.84, 17.26, 0, 70, 20.00, 0.00, 21, 0 };
                allData[164] = new double[] { 165, 122, 0.00, 3.08, 16.30, 0, 43, 22.13, 0.00, 16, 0 };
                allData[165] = new double[] { 166, 132, 7.00, 3.20, 23.26, 0, 77, 23.64, 23.14, 49, 0 };
                allData[166] = new double[] { 167, 110, 12.16, 4.99, 28.56, 0, 44, 27.14, 21.60, 55, 1 };
                allData[167] = new double[] { 168, 160, 1.52, 8.12, 29.30, 1, 54, 25.87, 12.86, 43, 1 };
                allData[168] = new double[] { 169, 126, 0.54, 4.39, 21.13, 1, 45, 25.99, 0.00, 25, 0 };
                allData[169] = new double[] { 170, 162, 5.30, 7.95, 33.58, 1, 58, 36.06, 8.23, 48, 0 };
                allData[170] = new double[] { 171, 194, 2.55, 6.89, 33.88, 1, 69, 29.33, 0.00, 41, 0 };
                allData[171] = new double[] { 172, 118, 0.75, 2.58, 20.25, 0, 59, 24.46, 0.00, 32, 0 };
                allData[172] = new double[] { 173, 124, 0.00, 4.79, 34.71, 0, 49, 26.09, 9.26, 47, 0 };
                allData[173] = new double[] { 174, 160, 0.00, 2.42, 34.46, 0, 48, 29.83, 1.03, 61, 0 };
                allData[174] = new double[] { 175, 128, 0.00, 2.51, 29.35, 1, 53, 22.05, 1.37, 62, 0 };
                allData[175] = new double[] { 176, 122, 4.00, 5.24, 27.89, 1, 45, 26.52, 0.00, 61, 1 };
                allData[176] = new double[] { 177, 132, 2.00, 2.70, 21.57, 1, 50, 27.95, 9.26, 37, 0 };
                allData[177] = new double[] { 178, 120, 0.00, 2.42, 16.66, 0, 46, 20.16, 0.00, 17, 0 };
                allData[178] = new double[] { 179, 128, 0.04, 8.22, 28.17, 0, 65, 26.24, 11.73, 24, 0 };
                allData[179] = new double[] { 180, 108, 15.00, 4.91, 34.65, 0, 41, 27.96, 14.40, 56, 0 };
                allData[180] = new double[] { 181, 166, 0.00, 4.31, 34.27, 0, 45, 30.14, 13.27, 56, 0 };
                allData[181] = new double[] { 182, 152, 0.00, 6.06, 41.05, 1, 51, 40.34, 0.00, 51, 0 };
                allData[182] = new double[] { 183, 170, 4.20, 4.67, 35.45, 1, 50, 27.14, 7.92, 60, 1 };
                allData[183] = new double[] { 184, 156, 4.00, 2.05, 19.48, 1, 50, 21.48, 27.77, 39, 1 };
                allData[184] = new double[] { 185, 116, 8.00, 6.73, 28.81, 1, 41, 26.74, 40.94, 48, 1 };
                allData[185] = new double[] { 186, 122, 4.40, 3.18, 11.59, 1, 59, 21.94, 0.00, 33, 1 };
                allData[186] = new double[] { 187, 150, 20.00, 6.40, 35.04, 0, 53, 28.88, 8.33, 63, 0 };
                allData[187] = new double[] { 188, 129, 2.15, 5.17, 27.57, 0, 52, 25.42, 2.06, 39, 0 };
                allData[188] = new double[] { 189, 134, 4.80, 6.58, 29.89, 1, 55, 24.73, 23.66, 63, 0 };
                allData[189] = new double[] { 190, 126, 0.00, 5.98, 29.06, 1, 56, 25.39, 11.52, 64, 1 };
                allData[190] = new double[] { 191, 142, 0.00, 3.72, 25.68, 0, 48, 24.37, 5.25, 40, 1 };
                allData[191] = new double[] { 192, 128, 0.70, 4.90, 37.42, 1, 72, 35.94, 3.09, 49, 1 };
                allData[192] = new double[] { 193, 102, 0.40, 3.41, 17.22, 1, 56, 23.59, 2.06, 39, 1 };
                allData[193] = new double[] { 194, 130, 0.00, 4.89, 25.98, 0, 72, 30.42, 14.71, 23, 0 };
                allData[194] = new double[] { 195, 138, 0.05, 2.79, 10.35, 0, 46, 21.62, 0.00, 18, 0 };
                allData[195] = new double[] { 196, 138, 0.00, 1.96, 11.82, 1, 54, 22.01, 8.13, 21, 0 };
                allData[196] = new double[] { 197, 128, 0.00, 3.09, 20.57, 0, 54, 25.63, 0.51, 17, 0 };
                allData[197] = new double[] { 198, 162, 2.92, 3.63, 31.33, 0, 62, 31.59, 18.51, 42, 0 };
                allData[198] = new double[] { 199, 160, 3.00, 9.19, 26.47, 1, 39, 28.25, 14.40, 54, 1 };
                allData[199] = new double[] { 200, 148, 0.00, 4.66, 24.39, 0, 50, 25.26, 4.03, 27, 0 };
                allData[200] = new double[] { 201, 124, 0.16, 2.44, 16.67, 0, 65, 24.58, 74.91, 23, 0 };
                allData[201] = new double[] { 202, 136, 3.15, 4.37, 20.22, 1, 59, 25.12, 47.16, 31, 1 };
                allData[202] = new double[] { 203, 134, 2.75, 5.51, 26.17, 0, 57, 29.87, 8.33, 33, 0 };
                allData[203] = new double[] { 204, 128, 0.73, 3.97, 23.52, 0, 54, 23.81, 19.20, 64, 0 };
                allData[204] = new double[] { 205, 122, 3.20, 3.59, 22.49, 1, 45, 24.96, 36.17, 58, 0 };
                allData[205] = new double[] { 206, 152, 3.00, 4.64, 31.29, 0, 41, 29.34, 4.53, 40, 0 };
                allData[206] = new double[] { 207, 162, 0.00, 5.09, 24.60, 1, 64, 26.71, 3.81, 18, 0 };
                allData[207] = new double[] { 208, 124, 4.00, 6.65, 30.84, 1, 54, 28.40, 33.51, 60, 0 };
                allData[208] = new double[] { 209, 136, 5.80, 5.90, 27.55, 0, 65, 25.71, 14.40, 59, 0 };
                allData[209] = new double[] { 210, 136, 8.80, 4.26, 32.03, 1, 52, 31.44, 34.35, 60, 0 };
                allData[210] = new double[] { 211, 134, 0.05, 8.03, 27.95, 0, 48, 26.88, 0.00, 60, 0 };
                allData[211] = new double[] { 212, 122, 1.00, 5.88, 34.81, 1, 69, 31.27, 15.94, 40, 1 };
                allData[212] = new double[] { 213, 116, 3.00, 3.05, 30.31, 0, 41, 23.63, 0.86, 44, 0 };
                allData[213] = new double[] { 214, 132, 0.00, 0.98, 21.39, 0, 62, 26.75, 0.00, 53, 0 };
                allData[214] = new double[] { 215, 134, 0.00, 2.40, 21.11, 0, 57, 22.45, 1.37, 18, 0 };
                allData[215] = new double[] { 216, 160, 7.77, 8.07, 34.80, 0, 64, 31.15, 0.00, 62, 1 };
                allData[216] = new double[] { 217, 180, 0.52, 4.23, 16.38, 0, 55, 22.56, 14.77, 45, 1 };
                allData[217] = new double[] { 218, 124, 0.81, 6.16, 11.61, 0, 35, 21.47, 10.49, 26, 0 };
                allData[218] = new double[] { 219, 114, 0.00, 4.97, 9.69, 0, 26, 22.60, 0.00, 25, 0 };
                allData[219] = new double[] { 220, 208, 7.40, 7.41, 32.03, 0, 50, 27.62, 7.85, 57, 0 };
                allData[220] = new double[] { 221, 138, 0.00, 3.14, 12.00, 0, 54, 20.28, 0.00, 16, 0 };
                allData[221] = new double[] { 222, 164, 0.50, 6.95, 39.64, 1, 47, 41.76, 3.81, 46, 1 };
                allData[222] = new double[] { 223, 144, 2.40, 8.13, 35.61, 0, 46, 27.38, 13.37, 60, 0 };
                allData[223] = new double[] { 224, 136, 7.50, 7.39, 28.04, 1, 50, 25.01, 0.00, 45, 1 };
                allData[224] = new double[] { 225, 132, 7.28, 3.52, 12.33, 0, 60, 19.48, 2.06, 56, 0 };
                allData[225] = new double[] { 226, 143, 5.04, 4.86, 23.59, 0, 58, 24.69, 18.72, 42, 0 };
                allData[226] = new double[] { 227, 112, 4.46, 7.18, 26.25, 1, 69, 27.29, 0.00, 32, 1 };
                allData[227] = new double[] { 228, 134, 10.00, 3.79, 34.72, 0, 42, 28.33, 28.80, 52, 1 };
                allData[228] = new double[] { 229, 138, 2.00, 5.11, 31.40, 1, 49, 27.25, 2.06, 64, 1 };
                allData[229] = new double[] { 230, 188, 0.00, 5.47, 32.44, 1, 71, 28.99, 7.41, 50, 1 };
                allData[230] = new double[] { 231, 110, 2.35, 3.36, 26.72, 1, 54, 26.08, 109.80, 58, 1 };
                allData[231] = new double[] { 232, 136, 13.20, 7.18, 35.95, 0, 48, 29.19, 0.00, 62, 0 };
                allData[232] = new double[] { 233, 130, 1.75, 5.46, 34.34, 0, 53, 29.42, 0.00, 58, 1 };
                allData[233] = new double[] { 234, 122, 0.00, 3.76, 24.59, 0, 56, 24.36, 0.00, 30, 0 };
                allData[234] = new double[] { 235, 138, 0.00, 3.24, 27.68, 0, 60, 25.70, 88.66, 29, 0 };
                allData[235] = new double[] { 236, 130, 18.00, 4.13, 27.43, 0, 54, 27.44, 0.00, 51, 1 };
                allData[236] = new double[] { 237, 126, 5.50, 3.78, 34.15, 0, 55, 28.85, 3.18, 61, 0 };
                allData[237] = new double[] { 238, 176, 5.76, 4.89, 26.10, 1, 46, 27.30, 19.44, 57, 0 };
                allData[238] = new double[] { 239, 122, 0.00, 5.49, 19.56, 0, 57, 23.12, 14.02, 27, 0 };
                allData[239] = new double[] { 240, 124, 0.00, 3.23, 9.64, 0, 59, 22.70, 0.00, 16, 0 };
                allData[240] = new double[] { 241, 140, 5.20, 3.58, 29.26, 0, 70, 27.29, 20.17, 45, 1 };
                allData[241] = new double[] { 242, 128, 6.00, 4.37, 22.98, 1, 50, 26.01, 0.00, 47, 0 };
                allData[242] = new double[] { 243, 190, 4.18, 5.05, 24.83, 0, 45, 26.09, 82.85, 41, 0 };
                allData[243] = new double[] { 244, 144, 0.76, 10.53, 35.66, 0, 63, 34.35, 0.00, 55, 1 };
                allData[244] = new double[] { 245, 126, 4.60, 7.40, 31.99, 1, 57, 28.67, 0.37, 60, 1 };
                allData[245] = new double[] { 246, 128, 0.00, 2.63, 23.88, 0, 45, 21.59, 6.54, 57, 0 };
                allData[246] = new double[] { 247, 136, 0.40, 3.91, 21.10, 1, 63, 22.30, 0.00, 56, 1 };
                allData[247] = new double[] { 248, 158, 4.00, 4.18, 28.61, 1, 42, 25.11, 0.00, 60, 0 };
                allData[248] = new double[] { 249, 160, 0.60, 6.94, 30.53, 0, 36, 25.68, 1.42, 64, 0 };
                allData[249] = new double[] { 250, 124, 6.00, 5.21, 33.02, 1, 64, 29.37, 7.61, 58, 1 };
                allData[250] = new double[] { 251, 158, 6.17, 8.12, 30.75, 0, 46, 27.84, 92.62, 48, 0 };
                allData[251] = new double[] { 252, 128, 0.00, 6.34, 11.87, 0, 57, 23.14, 0.00, 17, 0 };
                allData[252] = new double[] { 253, 166, 3.00, 3.82, 26.75, 0, 45, 20.86, 0.00, 63, 1 };
                allData[253] = new double[] { 254, 146, 7.50, 7.21, 25.93, 1, 55, 22.51, 0.51, 42, 0 };
                allData[254] = new double[] { 255, 161, 9.00, 4.65, 15.16, 1, 58, 23.76, 43.20, 46, 0 };
                allData[255] = new double[] { 256, 164, 13.02, 6.26, 29.38, 1, 47, 22.75, 37.03, 54, 1 };
                allData[256] = new double[] { 257, 146, 5.08, 7.03, 27.41, 1, 63, 36.46, 24.48, 37, 1 };
                allData[257] = new double[] { 258, 142, 4.48, 3.57, 19.75, 1, 51, 23.54, 3.29, 49, 0 };
                allData[258] = new double[] { 259, 138, 12.00, 5.13, 28.34, 0, 59, 24.49, 32.81, 58, 1 };
                allData[259] = new double[] { 260, 154, 1.80, 7.13, 34.04, 1, 52, 35.51, 39.36, 44, 0 };
                allData[260] = new double[] { 261, 118, 0.00, 2.39, 12.13, 0, 49, 18.46, 0.26, 17, 1 };
                allData[261] = new double[] { 263, 124, 0.61, 2.69, 17.15, 1, 61, 22.76, 11.55, 20, 0 };
                allData[262] = new double[] { 264, 124, 1.04, 2.84, 16.42, 1, 46, 20.17, 0.00, 61, 0 };
                allData[263] = new double[] { 265, 136, 5.00, 4.19, 23.99, 1, 68, 27.80, 25.86, 35, 0 };
                allData[264] = new double[] { 266, 132, 9.90, 4.63, 27.86, 1, 46, 23.39, 0.51, 52, 1 };
                allData[265] = new double[] { 267, 118, 0.12, 1.96, 20.31, 0, 37, 20.01, 2.42, 18, 0 };
                allData[266] = new double[] { 268, 118, 0.12, 4.16, 9.37, 0, 57, 19.61, 0.00, 17, 0 };
                allData[267] = new double[] { 269, 134, 12.00, 4.96, 29.79, 0, 53, 24.86, 8.23, 57, 0 };
                allData[268] = new double[] { 270, 114, 0.10, 3.95, 15.89, 1, 57, 20.31, 17.14, 16, 0 };
                allData[269] = new double[] { 271, 136, 6.80, 7.84, 30.74, 1, 58, 26.20, 23.66, 45, 1 };
                allData[270] = new double[] { 272, 130, 0.00, 4.16, 39.43, 1, 46, 30.01, 0.00, 55, 1 };
                allData[271] = new double[] { 273, 136, 2.20, 4.16, 38.02, 0, 65, 37.24, 4.11, 41, 1 };
                allData[272] = new double[] { 274, 136, 1.36, 3.16, 14.97, 1, 56, 24.98, 7.30, 24, 0 };
                allData[273] = new double[] { 275, 154, 4.20, 5.59, 25.02, 0, 58, 25.02, 1.54, 43, 0 };
                allData[274] = new double[] { 276, 108, 0.80, 2.47, 17.53, 0, 47, 22.18, 0.00, 55, 1 };
                allData[275] = new double[] { 277, 136, 8.80, 4.69, 36.07, 1, 38, 26.56, 2.78, 63, 1 };
                allData[276] = new double[] { 278, 174, 2.02, 6.57, 31.90, 1, 50, 28.75, 11.83, 64, 1 };
                allData[277] = new double[] { 279, 124, 4.25, 8.22, 30.77, 0, 56, 25.80, 0.00, 43, 0 };
                allData[278] = new double[] { 280, 114, 0.00, 2.63, 9.69, 0, 45, 17.89, 0.00, 16, 0 };
                allData[279] = new double[] { 281, 118, 0.12, 3.26, 12.26, 0, 55, 22.65, 0.00, 16, 0 };
                allData[280] = new double[] { 282, 106, 1.08, 4.37, 26.08, 0, 67, 24.07, 17.74, 28, 1 };
                allData[281] = new double[] { 283, 146, 3.60, 3.51, 22.67, 0, 51, 22.29, 43.71, 42, 0 };
                allData[282] = new double[] { 284, 206, 0.00, 4.17, 33.23, 0, 69, 27.36, 6.17, 50, 1 };
                allData[283] = new double[] { 285, 134, 3.00, 3.17, 17.91, 0, 35, 26.37, 15.12, 27, 0 };
                allData[284] = new double[] { 286, 148, 15.00, 4.98, 36.94, 1, 72, 31.83, 66.27, 41, 1 };
                allData[285] = new double[] { 287, 126, 0.21, 3.95, 15.11, 0, 61, 22.17, 2.42, 17, 0 };
                allData[286] = new double[] { 288, 134, 0.00, 3.69, 13.92, 0, 43, 27.66, 0.00, 19, 0 };
                allData[287] = new double[] { 289, 134, 0.02, 2.80, 18.84, 0, 45, 24.82, 0.00, 17, 0 };
                allData[288] = new double[] { 290, 123, 0.05, 4.61, 13.69, 0, 51, 23.23, 2.78, 16, 0 };
                allData[289] = new double[] { 291, 112, 0.60, 5.28, 25.71, 0, 55, 27.02, 27.77, 38, 1 };
                allData[290] = new double[] { 292, 112, 0.00, 1.71, 15.96, 0, 42, 22.03, 3.50, 16, 0 };
                allData[291] = new double[] { 293, 101, 0.48, 7.26, 13.00, 0, 50, 19.82, 5.19, 16, 0 };
                allData[292] = new double[] { 294, 150, 0.18, 4.14, 14.40, 0, 53, 23.43, 7.71, 44, 0 };
                allData[293] = new double[] { 295, 170, 2.60, 7.22, 28.69, 1, 71, 27.87, 37.65, 56, 1 };
                allData[294] = new double[] { 296, 134, 0.00, 5.63, 29.12, 0, 68, 32.33, 2.02, 34, 0 };
                allData[295] = new double[] { 297,142,0.00,4.19,18.04,0,56,23.65,20.78,42,1};
                allData[296] = new double[] { 298, 132, 0.10, 3.28, 10.73, 0, 73, 20.42, 0.00, 17, 0 };
                allData[297] = new double[] { 299, 136, 0.00, 2.28, 18.14, 0, 55, 22.59, 0.00, 17, 0 };
                allData[298] = new double[] { 300, 132, 12.00, 4.51, 21.93, 0, 61, 26.07, 64.80, 46, 1 };
                allData[299] = new double[] { 301, 166, 4.10, 4.00, 34.30, 1, 32, 29.51, 8.23, 53, 0 };
                
                double[] cSum = new double[9];
                double[] cAve = new double[9];
                double[] dSum= new double[9]; 
                double[] std = new double[9]; 
                double[] cMax = new double[9]{166, 4.10, 4.00, 34.30, 1, 32, 29.51, 8.23, 53}; 
                double[] cMin = new double[9]{166, 4.10, 4.00, 34.30, 1, 32, 29.51, 8.23, 53}; 
                  
                //double[][] newData = new double[300][];
                for (int i=0; i <=299; i++){
                    for (int j=0; j<=9; j++){
                        allData[i][j] = allData[i][j+1];
                    }
                }


                //make min.max.sum.std
                for(int col = 0; col <= 8; col++){
                    for(int row = 0; row <= 299; row++){
                        if(allData[row][col]<cMin[col]){
                            cMin[col] = allData[row][col];
                        }if(allData[row][col]>cMax[col]){
                            cMax[col] = allData[row][col];
                        }
                        cSum[col] += allData[row][col];
                        dSum[col] += (Math.Pow((allData[row][col] - cAve[col]), 2)/300);
                    }
                    cAve[col] = cSum[col]/300;
                    std[col] = Math.Pow(dSum[col],0.5);


                    Console.WriteLine("//////////////////////AVERAGE:  "+cAve[col]);
                    Console.WriteLine("//////////////////////stand:  "+std[col]);
                    Console.WriteLine("//////////////////////max:  "+cMax[col]);
                    Console.WriteLine("//////////////////////min:  "+cMin[col]);
                }




                //make data in standarlization
                /* 
                for (int i=0; i <=299; i++){
                    for (int j=0; j<=8; j++){
                        allData[i][j] =((allData[i][j]-cAve[j])/std[j]);
                        //Console.WriteLine("//////////////////////data:  "+allData[i][j]);
                    }
                }*/
                

                //make data in -10~10
                for (int i=0; i <=299; i++){
                    for (int j=0; j<=8; j++){
                        allData[i][j] =(20*(allData[i][j]-cMin[j])/(cMax[j]-cMin[j])-10);
                        double a = (allData[i][j]);
                        double b = (cMin[j]);
                    }
                }



                Console.WriteLine("Data generation weights:");
                ShowVector(weights, 4, 10, true);

                Console.WriteLine("\nFirst few lines of all data are (last column is the label): \n");
                ShowMatrix(allData, 4, 4, true);

                return allData;

        }




        //////////////////////////////////////////////////////////////////////////////////////////////////

        // generate artificial observations
        static double[][] MakeAllData(int numFeatures, int numRows, int seed)
        {
            Random rnd = new Random(seed);

            // numfeatures weights (bills)
            double[] weights = new double[numFeatures + 1]; // inc. b0
            for (int i = 0; i < weights.Length; ++i)
                weights[i] = 20.0 * rnd.NextDouble() - 10.0; // [-10.0 to +10.0]

            // numRows observations (congressmen voting on each bill) 
            // Last column reserved for label, which is categorical binary
            double[][] result = new double[numRows][]; // allocate matrix
            for (int i = 0; i < numRows; ++i)
                result[i] = new double[numFeatures + 1]; // Y in last column


            // generate random observations
            for (int i = 0; i < numRows; ++i) // for each row
            {
                double y = weights[0]; // the b0 
                for (int j = 0; j < numFeatures; ++j) // each feature / column except last
                {
                    double x = 20.0 * rnd.NextDouble() - 10.0; // random X in [10.0, +10.0]
                    result[i][j] = x; // store x
                    
                    double wx = x * weights[j + 1]; // weight * x 
                    y += wx; // accumulate to get Y
                    // now add some noise
                    y += numFeatures * rnd.NextDouble();
                }
                if (y > numFeatures) // because the noise was +, make it harder to be a 1
                    result[i][numFeatures] = 1.0; // store y in last column
                else
                    result[i][numFeatures] = 0.0;
            }
            Console.WriteLine("Data generation weights:");
            ShowVector(weights, 4, 10, true);

            Console.WriteLine("\nFirst few lines of all data are (last column is the label): \n");
            ShowMatrix(result, 4, 4, true);

            return result;
        }

                public static void ShowVector(double[] vector, int decimals, int lineLen, bool newLine)
        {
            for (int i = 0; i < vector.Length; ++i)
            {
                if (i > 0 && i % lineLen == 0) Console.WriteLine("");
                if (vector[i] >= 0) Console.Write(" ");
                Console.Write(vector[i].ToString("F" + decimals) + " ");
            }
            if (newLine == true)
                Console.WriteLine("");
        }

        static void ShowMatrix(double[][] matrix, int numRows, int decimals, bool indices)
        {
            for (int i = 0; i < numRows; ++i)
            {
                if (indices == true)
                    Console.Write("[" + i.ToString().PadLeft(2) + "]   ");
                for (int j = 0; j < matrix[i].Length; ++j)
                {
                    Console.Write(matrix[i][j].ToString("F" + decimals) + " ");
                }
                Console.WriteLine("");
            }
            int lastIndex = matrix.Length - 1;
            if (indices == true)
                Console.Write("[" + lastIndex.ToString().PadLeft(2) + "]   ");
            for (int j = 0; j < matrix[lastIndex].Length; ++j)
                Console.Write(matrix[lastIndex][j].ToString("F" + decimals) + " ");
            Console.WriteLine("\n");
        }

        static void MakeTrainTest(double[][] allData, int seed, out double[][] trainData, out double[][] testData)
        {
            Random rnd = new Random(seed);
            int totRows = allData.Length;
            int numTrainRows = (int)(totRows * 0.80); // 80% hard-coded
            int numTestRows = totRows - numTrainRows;
            trainData = new double[numTrainRows][];
            testData = new double[numTestRows][];

            double[][] copy = new double[allData.Length][]; // ref copy of all data
            for (int i = 0; i < copy.Length; ++i)
                copy[i] = allData[i];

            for (int i = 0; i < copy.Length; ++i) // scramble order
            {
                int r = rnd.Next(i, copy.Length); // use Fisher-Yates
                double[] tmp = copy[r];
                copy[r] = copy[i];
                copy[i] = tmp;
            }
            for (int i = 0; i < numTrainRows; ++i)
                trainData[i] = copy[i];

            for (int i = 0; i < numTestRows; ++i)
                testData[i] = copy[i + numTrainRows];
        }


            public class LogisticClassifier
    {
        private int numFeatures; // number of independent variables aka features
        private double[] weights; // b0 = constant
        private Random rnd;

        public LogisticClassifier(int numFeatures)
        {
            this.numFeatures = numFeatures; // number features/predictors
            this.weights = new double[numFeatures + 1]; // [0] = b0 constant
            this.rnd = new Random(0); // seed could be a parmeter to ctor
        }

        public double FindGoodL1Weight(double[][] trainData, int seed)
        {
            double result = 0.0;
            double bestErr = double.MaxValue;
            double currErr = double.MaxValue;
            double[] candidates = new double[] { 0.000, 0.001, 0.005, 0.010, 0.020, 0.050, 0.100, 0.150 };
            int maxEpochs = 1000;

            LogisticClassifier c = new LogisticClassifier(this.numFeatures);

            for (int trial = 0; trial < candidates.Length; ++trial)
            {
                double alpha1 = candidates[trial];
                double[] wts = c.TrainWeights(trainData, maxEpochs, seed, alpha1, 0.0);
                currErr = Error(trainData, wts, 0.0, 0.0);
                if (currErr < bestErr)
                {
                    bestErr = currErr;
                    result = candidates[trial];
                }
            }
            return result;
        }

        public double FindGoodL2Weight(double[][] trainData, int seed)
        {
            double result = 0.0;
            double bestErr = double.MaxValue;
            double currErr = double.MaxValue;
            double[] candidates = new double[] { 0.000, 0.001, 0.005, 0.010, 0.020, 0.050, 0.100, 0.150 };
            int maxEpochs = 1000;

            LogisticClassifier c = new LogisticClassifier(this.numFeatures);

            for (int trial = 0; trial < candidates.Length; ++trial)
            {
                double alpha2 = candidates[trial];
                double[] wts = c.TrainWeights(trainData, maxEpochs, seed, 0.0, alpha2);
                currErr = Error(trainData, wts, 0.0, 0.0);
                if (currErr < bestErr)
                {
                    bestErr = currErr;
                    result = candidates[trial];
                }
            }
            return result;
        }


            public double[] TrainWeights(double[][] trainData, int maxEpochs, int seed, double alpha1, double alpha2)
        {
            // use PSO. particle position == LR weights
            int numParticles = 10;
            double probDeath = 0.005;
            int dim = this.numFeatures; // need one wt for each feature, plus the b0 constant

            Random rnd = new Random(seed);

            int epoch = 0;
            double minX = -10.0; // for each weight. assumes data has been normalized about 0
            double maxX = 10.0;
            double w = 0.729; // inertia weight
            double c1 = 1.49445; // cognitive/local weight
            double c2 = 1.49445; // social/global weight
            double r1, r2; // cognitive and social randomizations

            Particle[] swarm = new Particle[numParticles];
            // best solution found by any particle in the swarm. implicit initialization to all 0.0
            double[] bestSwarmPosition = new double[dim];
            double bestSwarmError = double.MaxValue; // smaller values better

            // initialize each Particle in the swarm with random positions and velocities
            for (int i = 0; i < swarm.Length; ++i)
            {
                double[] randomPosition = new double[dim];
                for (int j = 0; j < randomPosition.Length; ++j)
                    randomPosition[j] = (maxX - minX) * rnd.NextDouble() + minX;

                // randomPosition is a set of weights
                double error = Error(trainData, randomPosition, alpha1, alpha2);
                double[] randomVelocity = new double[dim];
                for (int j = 0; j < randomVelocity.Length; ++j)
                {
                    double lo = 0.1 * minX;
                    double hi = 0.1 * maxX;
                    randomVelocity[j] = (hi - lo) * rnd.NextDouble() + lo;
                }
                swarm[i] = new Particle(randomPosition, error, randomVelocity,
                    randomPosition, error); // last two are best-position and best-error

                // does current Particle have global best position?
                // best position for the particle is the one that's closest to the label (Y)
                if (swarm[i].error < bestSwarmError)
                {
                    bestSwarmError = swarm[i].error;
                    swarm[i].position.CopyTo(bestSwarmPosition, 0);
                }
            }

            // main PSO algorithm
            int[] sequence = new int[numParticles]; // process particles in random order
            for (int i = 0; i < sequence.Length; ++i)
                sequence[i] = i;

            while (epoch < maxEpochs)
            {
                double[] newVelocity = new double[dim]; // step 1
                double[] newPosition = new double[dim]; // step 2
                double newError; // step 3

                Shuffle(sequence); // move particles in random sequence

                for (int pi = 0; pi < swarm.Length; ++pi) // each Particle (index)
                {
                    int i = sequence[pi];
                    Particle currP = swarm[i]; // for coding convenience

                    // 1. compute new velocity
                    for (int j = 0; j < currP.velocity.Length; ++j) // each x value of the velocity
                    {
                        r1 = rnd.NextDouble();
                        r2 = rnd.NextDouble();

                        // velocity depends on old velocity, best position of particle, and 
                        // best position of any particle
                        newVelocity[j] = (w * currP.velocity[j]) +
                            (c1 * r1 * (currP.bestPosition[j] - currP.position[j])) +
                            (c2 * r2 * (bestSwarmPosition[j] - currP.position[j]));
                    }

                    newVelocity.CopyTo(currP.velocity, 0);

                    // 2. use new velocity to compute new position
                    for (int j = 0; j < currP.position.Length; ++j)
                    {
                        newPosition[j] = currP.position[j] + newVelocity[j];  // compute new position
                        if (newPosition[j] < minX) // keep in range
                            newPosition[j] = minX;
                        else if (newPosition[j] > maxX)
                            newPosition[j] = maxX;
                    }

                    newPosition.CopyTo(currP.position, 0);

                    // 3. use new position to compute new error
                    newError = Error(trainData, newPosition, alpha1, alpha2);
                    currP.error = newError;

                    if (newError < currP.bestError) // new particle best?
                    {
                        newPosition.CopyTo(currP.bestPosition, 0);
                        currP.bestError = newError;
                    }

                    if (newError < bestSwarmError) // new swarm best?
                    {
                        newPosition.CopyTo(bestSwarmPosition, 0);
                        bestSwarmError = newError;
                    }

                    // 4. optional: does curr particle die?
                    double die = rnd.NextDouble();
                    if (die < probDeath)
                    {
                        // new position, leave velocity, update error
                        for (int j = 0; j < currP.position.Length; ++j)
                            currP.position[j] = (maxX - minX) * rnd.NextDouble() + minX;
                        currP.error = Error(trainData, currP.position, alpha1, alpha2);
                        currP.position.CopyTo(currP.bestPosition, 0);
                        currP.bestError = currP.error;

                        if (currP.error < bestSwarmError) // swarm best by chance?
                        {
                            bestSwarmError = currP.error;
                            currP.position.CopyTo(bestSwarmPosition, 0);
                        }
                    }

                } // each Particle
                ++epoch;
            } // while

            double[] retResult = new double[dim];
            Array.Copy(bestSwarmPosition, retResult, retResult.Length);
            return retResult;
        } // TrainWeights

        private void Shuffle(int[] sequence)
        {
            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = rnd.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }
        }

        public double Error(double[][] trainData, double[] weights, double alpha1, double alpha2)
        {
            // mean squared error using supplied weights
            // L1 regularization adds the sum of the absolute values of the weights
            // L2 regularization adds the sqrt of sum of squared values

            int yIndex = trainData[0].Length - 1; // y-value (0/1) is last column
            double sumSquaredError = 0.0;
            for (int i = 0; i < trainData.Length; ++i) // each data
            {
                double computed = ComputeY(trainData[i], weights);
                double desired = trainData[i][yIndex]; // ex: 0.0 or 1.0
                sumSquaredError += (computed - desired) * (computed - desired);
            }

            double sumAbsVals = 0.0; // L1 penalty
            for (int i = 0; i < weights.Length; ++i)
                sumAbsVals += Math.Abs(weights[i]);

            double sumSquaredVals = 0.0; // L2 penalty
            for (int i = 0; i < weights.Length; ++i)
                sumSquaredVals += (weights[i] * weights[i]);
            //double rootSum = Math.Sqrt(sumSquaredVals);

            return (sumSquaredError / trainData.Length) +
                (alpha1 * sumAbsVals) +
                (alpha2 * sumSquaredVals);
        }

        public double ComputeY(double[] dataItem, double[] weights)
        {
            double z = 0.0;

            z += weights[0]; // the b0 constant
            for (int i = 0; i < weights.Length - 1; ++i) // data might include Y
                z += (weights[i + 1] * dataItem[i]); // skip first weight

            return 1.0 / (1.0 + Math.Exp(-z));
        }

        public int ComputeDependent(double[] dataItem, double[] weights)
        {
            double sum = ComputeY(dataItem, weights);

            if (sum <= 0.5)
                return 0;
            else
                return 1;
        }

        public double Accuracy(double[][] trainData, double[] weights)
        {
            int numCorrect = 0;
            int numWrong = 0;
            int yIndex = trainData[0].Length - 1;
            for (int i = 0; i < trainData.Length; ++i)
            {
                double computed = ComputeDependent(trainData[i], weights); // implicit cast
                double desired = trainData[i][yIndex]; // 0.0 or 1.0

                double epsilon = 0.0000000001;
                if (Math.Abs(computed - desired) < epsilon)
                    ++numCorrect;
                else
                    ++numWrong;
            }
            return (numCorrect * 1.0) / (numWrong + numCorrect);
        }

        public class Particle // for PSO training
        {
            public double[] position; // equivalent to weights
            public double error; // measure of fitness
            public double[] velocity; // determines new position
            public double[] bestPosition; // best found by this Particle
            public double bestError;

            public Particle(double[] position, double error, double[] velocity,
            double[] bestPosition, double bestError)
            {
                this.position = new double[position.Length];
                position.CopyTo(this.position, 0);
                this.error = error;
                this.velocity = new double[velocity.Length];
                velocity.CopyTo(this.velocity, 0);
                this.bestPosition = new double[bestPosition.Length];
                bestPosition.CopyTo(this.bestPosition, 0);
                this.bestError = bestError;
            }
        } // (nested) class Particle





    } // LogisticClassifier
    }
}
