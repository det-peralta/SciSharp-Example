using ICSharpCode.SharpZipLib.GZip;
using ICSharpCode.SharpZipLib.Tar;
using Newtonsoft.Json;
using NumSharp;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;
using System.Net;
using System.Threading;
using System.Threading.Tasks;
using Tensorflow;
using static Tensorflow.Binding;

namespace predictor
{
    public class Helpers
    {
        public static string modelDir = "ssd_mobilenet_v1_coco_2018_01_28";
        public static string imageDir = "images";
        public static string pbFile = "frozen_inference_graph.pb";
        public static float MIN_SCORE = 0.5f;
        public static void PrepareData()
        {
            // get model file
            string url = "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz";
            Helpers.Download(url, modelDir, "ssd_mobilenet_v1_coco.tar.gz");

            Helpers.ExtractTGZ(Path.Join(modelDir, "ssd_mobilenet_v1_coco.tar.gz"), "./");

            // download sample picture
            url = $"https://github.com/tensorflow/models/raw/master/research/object_detection/test_images/image2.jpg";
            Helpers.Download(url, imageDir, "input.jpg");

            // download the pbtxt file
            url = $"https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt";
            Helpers.Download(url, modelDir, "mscoco_label_map.pbtxt");
        }
        public static void Predict()
        {
            // read in the input image
            var imgArr = Helpers.ReadTensorFromImageFile(Path.Join(imageDir, "input.jpg"));

            var graph = new Graph().as_default();
            graph.Import(Path.Join(modelDir, pbFile));

            using (var sess = tf.Session(graph))
            {
                Tensor tensorNum = graph.OperationByName("num_detections");
                Tensor tensorBoxes = graph.OperationByName("detection_boxes");
                Tensor tensorScores = graph.OperationByName("detection_scores");
                Tensor tensorClasses = graph.OperationByName("detection_classes");
                Tensor imgTensor = graph.OperationByName("image_tensor");
                Tensor[] outTensorArr = new Tensor[] { tensorNum, tensorBoxes, tensorScores, tensorClasses };

                var results = sess.run(outTensorArr, new FeedItem(imgTensor, imgArr));

                buildOutputImage(results);
            }
        }
        private static bool Download(string url, string destDir, string destFileName)
        {
            if (destFileName == null)
                destFileName = url.Split(Path.DirectorySeparatorChar).Last();

            Directory.CreateDirectory(destDir);

            string relativeFilePath = Path.Combine(destDir, destFileName);

            if (File.Exists(relativeFilePath))
            {
                Console.WriteLine($"{relativeFilePath} already exists.");
                return false;
            }

            var wc = new WebClient();
            Console.WriteLine($"Downloading {relativeFilePath}");
            var download = Task.Run(() => wc.DownloadFile(url, relativeFilePath));
            while (!download.IsCompleted)
            {
                Thread.Sleep(1000);
                Console.Write(".");
            }
            Console.WriteLine("");
            Console.WriteLine($"Downloaded {relativeFilePath}");

            return true;
        }
        private static void ExtractTGZ(String gzArchiveName, String destFolder)
        {
            var flag = gzArchiveName.Split(Path.DirectorySeparatorChar).Last().Split('.').First() + ".bin";
            if (File.Exists(Path.Combine(destFolder, flag))) return;

            Console.WriteLine($"Extracting.");
            var task = Task.Run(() =>
            {
                using (var inStream = File.OpenRead(gzArchiveName))
                {
                    using (var gzipStream = new GZipInputStream(inStream))
                    {
                        using (TarArchive tarArchive = TarArchive.CreateInputTarArchive(gzipStream))
                            tarArchive.ExtractContents(destFolder);
                    }
                }
            });

            while (!task.IsCompleted)
            {
                Thread.Sleep(200);
                Console.Write(".");
            }

            File.Create(Path.Combine(destFolder, flag));
            Console.WriteLine("");
            Console.WriteLine("Extracting is completed.");
        }
        private static NDArray ReadTensorFromImageFile(string file_name)
        {
            var graph = tf.Graph().as_default();

            var file_reader = tf.read_file(file_name, "file_reader");
            var decodeJpeg = tf.image.decode_jpeg(file_reader, channels: 3, name: "DecodeJpeg");
            var casted = tf.cast(decodeJpeg, TF_DataType.TF_UINT8);
            var dims_expander = tf.expand_dims(casted, 0);

            using (var sess = tf.Session(graph))
                return sess.run(dims_expander);
        }
        private static void buildOutputImage(NDArray[] resultArr)
        {
            // get pbtxt items
            PbtxtItems pbTxtItems = ParsePbtxtFile(Path.Join(modelDir, "mscoco_label_map.pbtxt"));

            // get bitmap
            Bitmap bitmap = new Bitmap(Path.Join(imageDir, "input.jpg"));

            var scores = resultArr[2].AsIterator<float>();
            var boxes = resultArr[1].GetData<float>();
            var id = np.squeeze(resultArr[3]).GetData<float>();
            for (int i = 0; i < scores.size; i++)
            {
                float score = scores.MoveNext();
                if (score > MIN_SCORE)
                {
                    float top = boxes[i * 4] * bitmap.Height;
                    float left = boxes[i * 4 + 1] * bitmap.Width;
                    float bottom = boxes[i * 4 + 2] * bitmap.Height;
                    float right = boxes[i * 4 + 3] * bitmap.Width;

                    Rectangle rect = new Rectangle()
                    {
                        X = (int)left,
                        Y = (int)top,
                        Width = (int)(right - left),
                        Height = (int)(bottom - top)
                    };

                    string name = pbTxtItems.items.Where(w => w.id == id[i]).Select(s => s.display_name).FirstOrDefault();

                    drawObjectOnBitmap(bitmap, rect, score, name);
                }
            }

            string path = Path.Join(imageDir, "output.jpg");
            bitmap.Save(path);
            Console.WriteLine($"Processed image is saved as {path}");
        }
        private static void drawObjectOnBitmap(Bitmap bmp, Rectangle rect, float score, string name)
        {
            using (Graphics graphic = Graphics.FromImage(bmp))
            {
                graphic.SmoothingMode = SmoothingMode.AntiAlias;

                using (Pen pen = new Pen(Color.Red, 2))
                {
                    graphic.DrawRectangle(pen, rect);

                    Point p = new Point(rect.Right + 5, rect.Top + 5);
                    string text = string.Format("{0}:{1}%", name, (int)(score * 100));
                    graphic.DrawString(text, new Font("Verdana", 8), Brushes.Red, p);
                }
            }
        }
        private static PbtxtItems ParsePbtxtFile(string filePath)
        {
            string line;
            string newText = "{\"items\":[";

            using (System.IO.StreamReader reader = new System.IO.StreamReader(filePath))
            {

                while ((line = reader.ReadLine()) != null)
                {
                    string newline = string.Empty;

                    if (line.Contains("{"))
                    {
                        newline = line.Replace("item", "").Trim();
                        //newText += line.Insert(line.IndexOf("=") + 1, "\"") + "\",";
                        newText += newline;
                    }
                    else if (line.Contains("}"))
                    {
                        newText = newText.Remove(newText.Length - 1);
                        newText += line;
                        newText += ",";
                    }
                    else
                    {
                        newline = line.Replace(":", "\":").Trim();
                        newline = "\"" + newline;// newline.Insert(0, "\"");
                        newline += ",";

                        newText += newline;
                    }

                }

                newText = newText.Remove(newText.Length - 1);
                newText += "]}";

                reader.Close();
            }

            PbtxtItems items = JsonConvert.DeserializeObject<PbtxtItems>(newText);

            return items;
        }

        private static void CaptureCameraCallback()
        {
            var frame = new Mat();
            var capture = new VideoCapture();
            capture.Open(2);
            int isCameraRunning = 0;
            while (isCameraRunning == 1)
            {
                capture.Read(frame);
                var image = BitmapConverter.ToBitmap(frame);
                //pictureBox1.Image = image;
                image = null;
            }
        }
    }

    public class PbtxtItem
    {
        public string name { get; set; }
        public int id { get; set; }
        public string display_name { get; set; }
    }
    public class PbtxtItems
    {
        public List<PbtxtItem> items { get; set; }
    }
}
