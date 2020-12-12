using System;
using System.Collections.Generic;
using System.Text;

namespace ObjectDetection.Utilities
{
   public struct ImageSettings
   {
      public const int ImageWidth = 416;
      public const int ImageHeight = 416;
   }
   public struct ONNXsettings
   {
      public const string InputName = "image";
      public const string OutputName= "grid";
   }
}
