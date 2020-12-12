using Microsoft.ML;
using ObjectDetection.Model;
using ObjectDetection.Utilities;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace ObjectDetection.Model
{
   public static class ConsumeModel
   {
      private static string MODEL_PATH = Path.Combine("Model.zip");
      private static Lazy<PredictionEngine<ModelInput, ModelOutput>> PredictionEngine;

      public static async Task<ModelOutput> PredictAsync(ModelInput modelInput, bool saveOnImage = false)
      {
         PredictionEngine = await CreatePredictionEngineAsync();
         modelInput.PredictedImagePath = "";
         ModelOutput probability = PredictionEngine.Value.Predict(modelInput);

         if (saveOnImage)
         {

            var image = await DrawBoundingBoxAsync(modelInput, probability);
            if (image != null)
               modelInput.PredictedImagePath = SaveOnImage(image, modelInput.ImagePath);
         }
         return probability;
      }

      private static string SaveOnImage(Image image, string inputImage)
      {
         var path = Path.Combine(Directory.GetCurrentDirectory(), "Images");
         if (!Directory.Exists(path))
            Directory.CreateDirectory(path);
         var imgFilename = Guid.NewGuid().ToString() + Path.GetExtension(inputImage);
         var fullPath = Path.Combine(path, imgFilename);

         image.Save(fullPath);
         return fullPath;
      }

      private static async Task<Lazy<PredictionEngine<ModelInput, ModelOutput>>> CreatePredictionEngineAsync()
      {
         MLContext context = new MLContext(0);
         ITransformer model;
         return await Task.Run(() =>
         {
            model = context.Model.Load(MODEL_PATH, out _);
            var engine = context.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);
            return new Lazy<PredictionEngine<ModelInput, ModelOutput>>(engine);
         });
      }

      private static async Task<Image> DrawBoundingBoxAsync(ModelInput modelInput, ModelOutput probability)
      {
         try
         {
            var image = Image.FromFile(modelInput.ImagePath);

            var orgHeight = image.Height;
            var orgWidth = image.Width;

            OutputParser parser = new OutputParser();

            var boundingBoxes = parser.ParseOutputs(probability.Scores);
            var filteredBoundingBoxes = new List<BoundingBox>();
            var boxLabels = boundingBoxes.Select(x => x.Label).Distinct().ToList();

            probability.Labels = boxLabels.ToArray();

            foreach (var label in boxLabels)
            {
               var singleLabelBoundingBox = boundingBoxes.Where(x => x.Label == label).ToList();
               filteredBoundingBoxes.AddRange(
                  parser.FilterBoundingBoxes(singleLabelBoundingBox, 5, .5F).ToList()
                  );
            }

            probability.Objects = filteredBoundingBoxes.Count;
            //foreach (var label in boxLabels)
            //{
            //   var singleLabelBoxes = boundingBoxes.Where(x => x.Label == label)
            //         .OrderByDescending(x => x.Confidence).ToList();

            //   foreach (var single in singleLabelBoxes)
            //   {
            //      var result = singleLabelBoxes
            //         .Where(x => (Math.Abs(x.Dimensions.X - single.Dimensions.X) < 5) || (Math.Abs(x.Dimensions.Y - single.Dimensions.Y) < 5)).ToList();

            //      result.Remove(single);

            //      foreach (var item in result)
            //      {
            //         boundingBoxes.Remove(item);
            //         singleLabelBoxes = boundingBoxes.Where(x => x.Label == label)
            //         .OrderByDescending(x => x.Confidence).ToList();
            //         singleLabelBoxes.Remove(single);
            //      }
            //   }
            //}
            if (filteredBoundingBoxes.Count > 0)
            {
               await Task.Run(() =>
               {
                  foreach (var box in filteredBoundingBoxes)
                  {
                     var x = (uint)Math.Max(box.Dimensions.X, 0);
                     var y = (uint)Math.Max(box.Dimensions.Y, 0);
                     var width = (uint)Math.Min(orgWidth - x, box.Dimensions.Width);
                     var height = (uint)Math.Min(orgHeight - y, box.Dimensions.Height);

                     x = (uint)orgWidth * x / ImageSettings.ImageWidth;
                     y = (uint)orgHeight * y / ImageSettings.ImageHeight;
                     width = (uint)orgWidth * width / ImageSettings.ImageWidth;
                     height = (uint)orgHeight * height / ImageSettings.ImageHeight;

                     string text = $"{box.Label} ({(box.Confidence * 100).ToString("0")}%)";

                     using (Graphics thumbnailGraphic = Graphics.FromImage(image))
                     {
                        thumbnailGraphic.CompositingQuality = CompositingQuality.HighQuality;
                        thumbnailGraphic.SmoothingMode = SmoothingMode.HighQuality;
                        thumbnailGraphic.InterpolationMode = InterpolationMode.HighQualityBicubic;

                        // Define Text Options
                        Font drawFont = new Font("Bahnschrift", 16, FontStyle.Bold);
                        SizeF size = thumbnailGraphic.MeasureString(text, drawFont);
                        SolidBrush fontBrush = new SolidBrush(Color.Black);
                        Point atPoint = new Point((int)x, (int)y - (int)size.Height - 1);

                        // Define BoundingBox options
                        Pen pen = new Pen(box.BoxColor, 3.2f);
                        SolidBrush colorBrush = new SolidBrush(box.BoxColor);

                        thumbnailGraphic.FillRectangle(colorBrush, (int)x, (int)(y - size.Height - 1), (int)size.Width, (int)size.Height);

                        thumbnailGraphic.DrawString(text, drawFont, fontBrush, atPoint);

                        // Draw bounding box on image
                        thumbnailGraphic.DrawRectangle(pen, x, y, width, height);
                     }
                  }
               });

            }
            return image;
         }
         catch (Exception ex)
         {
            Console.WriteLine(ex.Message);
            return null;
         }

      }
   }
}
