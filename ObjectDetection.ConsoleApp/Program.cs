using Microsoft.ML;
using ObjectDetection.Model;
using ObjectDetection.Utilities;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;

namespace ObjectDetection.ConsoleApp
{
   class Program
   {
      static void Main(string[] args)
      {
         CreateModel();
         var result = ConsumeModel.PredictAsync(new ModelInput() { ImagePath = "Samples\\Man-Dog.jpg" },true);
      }

      static void CreateModel()
      {
         MLContext context = new MLContext(0);

         IDataView dataView = context.Data.LoadFromEnumerable(new List<ModelInput>());

         IEstimator<ITransformer> trainingPipeLine = context.Transforms.LoadImages("Images", "", "ImagePath")
            .Append(context.Transforms.ResizeImages("Images", ImageSettings.ImageWidth, ImageSettings.ImageHeight))
            .Append(context.Transforms.ExtractPixels(ONNXsettings.InputName, "Images"))
            .Append(context.Transforms.ApplyOnnxModel(new string[] { ONNXsettings.OutputName }, new string[] { ONNXsettings.InputName }, "tinyyolov2-8.onnx"));

         ITransformer model = trainingPipeLine.Fit(dataView);

         context.Model.Save(model, dataView.Schema, "Model.zip");
      }
   }
}
