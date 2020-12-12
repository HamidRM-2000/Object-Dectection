using Microsoft.ML.Data;
using ObjectDetection.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ObjectDetection.Model
{
   public class ModelOutput
   {
      [ColumnName(ONNXsettings.OutputName)]
      public float[] Scores { get; set; }
      public string[] Labels { get; set; }
      public int Objects { get; set; }
   }
}
